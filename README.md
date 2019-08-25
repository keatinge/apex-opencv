# Context
The image you are looking at is a screenshot of a twitch.tv livestream of a game called Apex Legends. Essentially, this person is playing the game, and people are watching him play, live.
![livestream screenshot](https://i.imgur.com/EdL3YB3.jpg)

It is very common for livestreamers to then upload their best games to YouTube, where they can often get millions of views. The screenshot below is some Apex Legends gameplay that has nearly 2.2 million views. An interesting fact to note is that this game was taken directly from a livestream.
![youtube upload](https://i.imgur.com/Ud83ZW0.jpg)

# The Project
The goal of this project is to automate the processing of combing through hundreds of hours of livestreamed gameplay to detect the best games. We will define the best games as those where the player has ended with the highest number of kills. You can see the players kills in the top right of the screen, next to the skull icon.

There were 5 main components that needed to work to completely automate this process

1. Reverse engineer the Twitch protocol to locally download the entire livestream as a video file for further processing.
1. Identify at what points in that livestream did each game start and end so that we can clip out the game where the player did well.
1. For each game, identify the number of kills the player had at the end of the game.
1. Clip out the best game from the livestream so it can be easily uploaded to YouTube

## Reverse engineering the TwitchTV livestreaming protocol
Note: If you are more interested in the computer vision, machine learning, and pretty pictures you can [skip to the next section on postprocessing](#postprocessing-computer-vision-and-machine-learning)

### How a TwitchTV VOD is streamed
To keep things simple, this project does not process the livestream as it is happening. TwitchTV allows you to re-watch the entire contents of a previously completed livestream, called a VOD. This project processes VODS, this allows the project to process all past live streams as-well. A program that could only process livestreams as they were happening would be less useful and harder to debug.

Before we can automate the process of downloading a vod, we need to understand it. The following paragraphs will explain how Twitch delivers a vod from their servers to your browser.


After authentication, the first step to watching a VOD is a request to an m3u8 file. For example `https://usher.ttvnw.net/vod/468178925.m3u8?allow_source=true&....`

![m3u8 resolutions](https://i.imgur.com/ta6gvoT.png)


This yields a listing containing available resolutions. Each resolution has its own m3u8 file called `index-dvr.m3u8`, the link is found in the text of the m3u8 file (as you can see below).
```
#EXTM3U
#EXT-X-TWITCH-INFO:ORIGIN="s3",B="false",REGION="NA",USER-IP="209.217.198.34",SERVING-ID="dbc14278b8cd44a0be78dc0d49e41a3c",CLUSTER="metro_vod",USER-COUNTRY="US",MANIFEST-CLUSTER="metro_vod"
#EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID="chunked",NAME="1080p60",AUTOSELECT=YES,DEFAULT=YES
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=8144121,CODECS="avc1.4D402A,mp4a.40.2",RESOLUTION="1920x1080",VIDEO="chunked",FRAME-RATE=59.998
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/chunked/index-dvr.m3u8
#EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID="720p60",NAME="720p60",AUTOSELECT=YES,DEFAULT=YES
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=3076544,CODECS="avc1.4D401F,mp4a.40.2",RESOLUTION="1280x720",VIDEO="720p60",FRAME-RATE=59.998
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/720p60/index-dvr.m3u8
... (and other resolutions)
```

The `index-dvr.m3u8` then contains a list of ts files.

![ts files m3u8](https://i.imgur.com/cHHQWxk.png)

Each ts file is a 10 second video clip. When you are watching a livestream VOD, your browser is constantly downloading these 10 second clips and stitching them together to create the illusion of a continuous video. Watching an in-progress livestream works in much the same way, with the exception that the clips are no longer each a perfect 10 seconds, and you have to continuously request the same m3u8 file to get an updated listing of all the available ts files.

### Automating the process
Wtih an understanding of how the TwitchTV livestreaming protocol works, implementing it in python is straightforward.

First a request is made to the following url to recieve a `token` and `sig` which are used for authentication for the next request
```
https://api.twitch.tv/api/vods/{vod_id}/access_token?need_https=true&oauth_token=&platform=_&player_backend=mediaplayer&player_type=site
```


Next, a request is made to the first m3u8 file (url shown below), which will contain links to the other m3u8 files
```
https://usher.ttvnw.net/vod/{vod_id}.m3u8?allow_source=true&token={token}&sig={sig}
```

From that m3u8 file, we find the `index-dvr.m3u8` associated with the 1080p60fps stream and make a request to that url (an example is shown below) to figure out the ts files to download
```
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/chunked/index-dvr.m3u8
```

After that, we make requests to download every ts file.
```
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/chunked/1.ts
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/chunked/2.ts
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/chunked/3.ts
https://vod-metro.twitch.tv/07ab6883d55b556c1365_shroud_35322098080_1276304970/chunked/4.ts
```

And finally we combine all the ts files into a single large ts files. This can be done with simple file concatenation.

At this point we are done. We have successfully written a program which can take any vod id as it's input and can output a single ts file containing the entire livestream. This prepares us for the next step, processing the livestream.

# Postprocessing: computer vision, and machine learning
## Sampling frames
Because traditional computer vision techniques operate on images, not video, the first step is to extract a sequence of frames from the video. At 1080p60fps a 5 hour livestream would generate `5*60*60*60=1,080,000`frames, it would take terabytes to store all these frames as BMP files.

 Thankfully, there is no reason to process every single frame. The players score is not changing 60 times per second. Sampling every couple of seconds is more than enough. For my purposes I settled on .33 FPS or roughly 1 frame every 3 seconds. To reduce size I am also simultaneously cropping the frames while they are being exported. Most of the relevant information is found on the top half of the screen, so there is no need to save the rest.

 The easiest (and most perfomant) solution to exporting frames was to use ffmpeg.

 ```
 def sample_frames(src_video_file, output_loc):
    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg", "-i", src_video_file, "-vf", f"fps={SAMPLE_FPS},crop=in_w:0.4*in_h:0:0", "-nostdin", os.path.join(output_loc, "frame%04d.bmp")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
```

This gives a large directory containing thousands of BMP files. ffmpeg can output in many different formats, but BMP is the fastest to export, since it doesn't require any compression. Given the large number of frames which need to be both exported and then opened for processing, I was happy to trade disk space for decreased processing time.

![images](https://i.imgur.com/yjkckSZ.png)

## Finding game start and ends
Armed with 1 frame for every 3 seconds, these frames can then be processed to look for markers indiciating the start or end of a game. When loading into a game, there is a champion select sceen with large clear text that says `SELECT LEGEND`. Here's an example:

![select legend screen](https://i.imgur.com/P4aQ9mK.png)


The ending screen is similarly distinctive.
![ending screen](https://i.imgur.com/xQk9VnY.png)

[Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is a optical character recognition engine by google. The tesseract binary takes a path to an image and outputs the text it can detect in that image. This project uses the library `pytesseract` which is a thin wrapper around the tesseract binary. Tesseract works quite well when presented with clear large lines of text on a neutral background.

 To detect the start and ending screens, I send these images to Tesseract, and then process the text that tesseract has found in the image. The starting screen can be detecting by searching for the literal strings `SELECT LEGEND`. You may be thinking the ending screen can be detected by the literal string `SQUAD ELIMNINATED` but that would be a mistake, since if the player is to win the game the text would not say `SQUAD ELIMINATED`. Instead I found best results by looking for the `PLACED #n OF k` portion of text in the upper left.

The code to determine is a frame is a game start, end, or neither is amazingly simple thanks to the `pytesseract` library.

```
def get_frame_markers(full_im_path):
    im = Image.open(full_im_path)
    s = pytesseract.image_to_string(im)
    lower_text = s.lower()

    markers = {
        "is_game_start": "select legend" in lower_text,
        "is_game_end": ("placed" in lower_text and "of" in lower_text),
    }

    return markers
```



## Finding the players score at a specific frame
Determining the player's score in any frame is by far the most challenging component of this entire project. I will start by showing a few example frames to illustrate why this is difficult.

A player with 8 kills
![frame1](https://i.imgur.com/crElor2.png)
A player with 3 kills and 1 spectator
![frame2](https://i.imgur.com/sBLRArK.png)
A player with 0 kills
![frame3](https://i.imgur.com/HTUxHaZ.png)
A player that is opening their inventory, kills are not shown
![frame4](https://i.imgur.com/dEtRYNm.png)
A frame from when the players score has changed, causing it to appear red
![frame5](https://i.imgur.com/o2aLJoY.png)
The player is dead, instead of showing their score, the score of the person whom they are spectating is shown
![frame6](https://i.imgur.com/SsLtNuS.png)

Additional complexity is caused by the fact that the background near the score is not very clean. The first thing to notice is that the score's background depends on whatever the player happens to be looking at, meaning the background could be any color. Another issue is the black and white dot pattern to the right of the score, in the next section you'll see why this becomes a problem.
![close up](https://i.imgur.com/XyQKWX3.png)

### The score-detecting algorithm
I explored many different approaches to detecting the score and found best results with the following process.
1. Find the location of score by looking for the skull. The score is not always in the same place because the text to the right of it is not always the same size.
2. Extract the individual digits from the score by looking for connected components
3. Send each individual digit into a custom-trained convolutional neural net
4. Recombine the labeled digits into a single score


### Finding the score by looking for the skull
The first step in looking for the skull is to remove some noise from the image. A strategy that I found produced excellent results was to average the frame with the previous frame and the next frame. Keep in mind that the previous and next frames are actually 3 seconds apart, so they tend to be significantly different. We are exploiting the fact that we expect the score to not change very often, and this assumption generally holds true. This makes anything that did not remain constant between these 3 frames very blurry. Here's what this looks like for 50 random frames:

```
def mean_frame(streamdir, framenum, dist):
    sum_img = None
    count = 0
    for i in range(framenum-dist, framenum+dist+1):
        im = open_frame(streamdir, i)
        if im is None:
            continue
        im_sect = get_image_top_right_roi(im)
        count += 1
        if sum_img is None:
            sum_img = im_sect.astype(np.float)
        else:
            sum_img += im_sect

    assert sum_img is not None, "Couldnt open %s" % framenum
    return (sum_img/count).astype(np.uint8)
```

![50 frames mean](https://i.imgur.com/CzYKIgS.jpg)

Notice how the score remains very sharp while the rest of the image becomes blurred.

In the next step, aggressive thresholding is used to remove everything but the most bright parts of the image.
```
cv2.threshold(roi_gs, max(170, np.max(roi_gs) - 25), 255, cv2.THRESH_BINARY)
```
![roi threshold](https://i.imgur.com/G7BVIHe.png)


Median blurring is then applied to remove most of the noise, leaving us with only the large objects (the skull)
```
v2.medianBlur(roi_th1, 7)
 ```
![median blur](https://i.imgur.com/GC7bKJV.png)

Then `cv2.findContours` is used to find all the contours, shown below is the found contours plotted ontop of the original averaged image.
```
cv2.findContours(med, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
```
![contours](https://i.imgur.com/OnDFq8S.jpg)

Then we filter on the contours, removing all images that don't have any found contours and removing all images with contours too large or too small. This leaves us with the following set of reasonable images to work with and the exact location of the skulls within them.
![filtered contours](https://i.imgur.com/CCC1UXa.png)

### Extracting individual digits
Next, using the found skulls, we know the score is always going to be the exact same distance from the skull, so using some constants found through trial and error, we can generate a bounding box that the score must be inside of.

![score bounding box](https://i.imgur.com/k1lGwPo.png)

Here is what the extracted ROIs look like
![rois](https://i.imgur.com/RkueY3t.png)

We then run Otsu's binarization algorithm on the extracted ROIs to attempt to separate the digits from the background
```
cv2.threshold(kills_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
````
![roi thresh](https://i.imgur.com/u0aSArF.png)

Notice that the fourth image comes out very poorly. This is due to the inconsistent background behind the 2. The thresholded images are then checked for their proportion of dark pixels. If there are too many dark pixels, like in the fourth image, they are removed.
```
def get_binarized_kills_roi(kills_roi):
    ret, binarized = cv2.threshold(kills_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    on_pixels = (binarized == 0).sum()
    prop = on_pixels / binarized.size

    if not (.069 <= prop <= .26):
        return None

    return binarized
```

The constants here were found by looking at histograms on a very large number of images, for example:
![on pixels histogram](https://i.imgur.com/L1yL0BM.png)
As you can see, clearly anything below .069 or above .26 must have been an error and can safely be filtered out.

This filtering does indeed correctly remove the one problematic image, leaving us with 19 remaining images.
![histogram filtered roi thresh](https://i.imgur.com/sfp1s2L.png)

The next step is to find all connected components in each image, to separate out the individual digits.
```
labels, conn_comm = cv2.connectedComponents(255 - binarized_kills_roi)
```
![connected components](https://i.imgur.com/qMIdDQ1.png)

As you can see, many of the images have some small noise, resulting in extraneous connected components. Filtering by on pixels (this time the raw number of pixels rather than the proportion) and by the aspect ratio can correctly remove all the noise. This leaves us with the following digits.

![connected filtered](https://i.imgur.com/xXcBubY.png)

At this point you may be thinking the project is nearly finished. In theory I would just need to send these images to Tesseract to perform OCR and I will be able to determine the score at each frame. Unfortunately, as I had discovered, Tesseract performs very poorly on single digits. Here's how pytesseract labels these images

![pytesseract on digits](https://i.imgur.com/uh4vXm0.png)

Surely we can do much better than 40% correct?

Interestingly, Tesseract performs much better if some padding is image to the images

![pytesseract w/ padding](https://i.imgur.com/ZjuQeNF.png)

Though, on a larger dataset, there is clearly room for improvement. I count 16 mistakes out of 74 images, giving us only 78% accuracy.

![pytesseract on more](https://i.imgur.com/uoEihOd.png)


### The convolutional neural net
Given how normalized these digits are, I was certain I could do much better than 78% accuracy, so I began exploring using a convolutional neural net.

#### Generating labels
To train the CNN I was going to need a large number of labeled frames. Thankfully I've already written a program to download vods and extract the frames, but now I needed some way to record the labels. Additionally, given the sequential nature of the frames I should easily be able to label a large number of frames very quickly since I only need to find the points where the score changes.

To do the labeling, I built a simple web application to show me a large number of frames, where I could then mass label images.
![labeling](https://i.imgur.com/6C32wFT.gif)

After an hour of labeling I had 4956 labeled frames stored in a json file like this:
```
{
    "labels": [
        ...
        {
            "frame_num": 1451,
            "label": 3
        },
        {
            "frame_num": 1452,
            "label": 3
        },
        {
            "frame_num": 1453,
            "label": 4
        },
        {
            "frame_num": 1454,
            "label": 4
        },
        ...
    ]
}
```


#### The model
The idea of using a convolutional layer followed by dropout and flatten layers is known to perform very well on digit recognition, so I stuck with that basic model.

I found that the hyperparameters played a large role in how well the model could perform, so I spent significant time tuning the parameters. The number of convolutional kernels, the kernel size, the number of dense neurons, the dropout constant, the learning rate, the epochs, and the batch size were all found through grid search, evaluating a large number of different models on the same dataset.
```
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(25,25,1), data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=100, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0005),
              metrics=[metrics.categorical_accuracy])
history = model.fit(x_train, y_train, epochs=13, batch_size=16)
```

### Training the model
To train the model I followed the standard practice of splitting the data and using a portion for training and a portion for validating. This is to ensure the model doesn't "memorize" the correct answer. By validating the model on data it has never seen before, I can ensure the model will be able to generalize to unseen data.

Typically 80% of the data is used for training and 20% for validation, but I found the model really didn't need too much training data. I got just as good results using a 50% split, so I stuck with that to ensure I could adequately evaluate the effectiveness of the model.

Here's a plot of the loss (categorical cross entropy) plotted over time during training. Evidently there is a plateau around the 13th epoch, which is how I decided to end the training at 13 epochs, to avoid overfitting.

![training loss](https://i.imgur.com/ezuyUmj.png)

### Evaluating the model
To evaluate the model I used two main tools: `sklearn.metrics.classification_report` and `sklearn.metrics.confusion_matrix`. Here is what the classification report looks like, evaluating the model on 2478 digits which the model has never seen before.
```
              precision    recall  f1-score   support

           0    1.00000   1.00000   1.00000       171
           1    1.00000   0.99847   0.99924       655
           2    1.00000   0.99432   0.99715       352
           3    0.97834   1.00000   0.98905       271
           4    1.00000   0.99500   0.99749       200
           5    0.98980   1.00000   0.99487       291
           6    0.99180   0.99180   0.99180       122
           7    1.00000   0.99367   0.99683       158
           8    0.98519   0.99254   0.98885       134
           9    1.00000   0.95968   0.97942       124

    accuracy                        0.99516      2478
   macro avg    0.99451   0.99255   0.99347      2478
weighted avg    0.99523   0.99516   0.99515      2478
```


Both precision and recall are very good, indicating that the model is correctly determining the digits rather than just memorizing the distribution of digits. Though it was clear the model has some issues with detecting the digit 9. I suspect this could be due to the underlying distribution of digits in a players score. As you can see from the confusion matrix, 1 kill is very common, 2 kills is second most common, etc. This makes sense when you consider that before a player can get 2 kills, they must first get 1 kill.

A potential improvement would be to train the model on a more even distribution of digits, though that would require many more manually labeled frames, since 8 and 9 kills are relatively rare.

Below is the confusion matrix:
```
[[171   0   0   0   0   0   0   0   0   0]
 [  0 654   0   1   0   0   0   0   0   0]
 [  0   0 350   2   0   0   0   0   0   0]
 [  0   0   0 271   0   0   0   0   0   0]
 [  0   0   0   0 199   0   0   0   1   0]
 [  0   0   0   0   0 291   0   0   0   0]
 [  0   0   0   1   0   0 121   0   0   0]
 [  0   0   0   0   0   0   1 157   0   0]
 [  0   0   0   0   0   1   0   0 133   0]
 [  0   0   0   2   0   2   0   0   1 119]]
 ```


The actual values are along the x axis, while the predicted is along the y axis. For example, if you look at the last row, you are looking at the true 9s. From that last row we can see that when the digit is in fact a 9, the model sometimes misclassified it as a 3 or a 5. Here's what those misclassified images look like:
![missclassified](https://i.imgur.com/YCsBTCJ.png)

Keep in mind, we are testing the model on 2478 images, and it has correctly labeled all of them except 12. This is a massive improvement. 2466/2478 were correctly labeled, that's 99.515% accuracy.

Below is some images and what the model has determined the digit is, as you can see, it's almost always correct.
![performance](https://i.imgur.com/HFfX48A.png)

With this model, and the previous work of separating digits, we can now determine the number of kills on every frame
```
def classify_frame(stream_dir, frame_num, model):
    char_rois = diffing.get_digits_for_frame_on_bg(stream_dir, frame_num)
    if char_rois is None or len(char_rois) == 0:
        return None, None

    char_rois_for_xs = np.array(char_rois).reshape(-1, 25, 25, 1) #for keras
    predicted = one_hot_to_categorical(model.predict(char_rois_for_xs), list(range(10)))
    as_str = "".join(map(str, predicted))
    return int(as_str), char_rois
```

Here's what the finals results look like, using the model to determine the number of kills on 20 random frames.
![results](https://i.imgur.com/dw7tmK6.jpg)


## Putting it all together
With all the main pieces in place, what remains is to connect them up. The process works like this
1. Twitch livestream is downloaded and frames are produced
1. Frames are processed to find the start and end of games using pytesseract
1. Starting at the end of each game, iterate backgrounds through the frames collecting the player's kills at each frame until 20 samples are acquired.
1. The median of the samples is then used as the best estimate for the player score. The median of 20 samples is used to avoid having a couple misclassified digits affect the calculated score.
1. FFmpeg is used to clip out the single game from the large ts file, optionally converting to a video format more YouTube friendly.


## Lessons Learned
1. Determinism is very important in debugging. The program had initially downloaded a livestream as it was happening, but this made one-off issues very difficult to debug. Development was much simpler after switching to downloading completed livestreams rather than downloading livestreams while they were occurring.
1. When reading and writing thousands of images, avoding compression can result in large speedups. I initially had ffmpeg exporting PNGs instead of BMPs. This made converting the video into individual frames significantly slower. Additionally, every time I was processing these frames they would have to be uncompressed before processing. Switching everything to BMPs was a valuable trade-off. Processing time was much more important than disk usage during development.
1. Tuning hyperparameters can have a significant impact on the effectiveness of the convolutional neural net. Although it was a tedious and time-consuming processes, it was definitely worthwhile to take the time to investigate which parameters performed best.
1. Changes should be evaluated on a large number of images. Initially during development I would notice some problematic frames and tune the algorithm to perform better on those frames. This would then unknowingly make the algorithm perform worse on different types of frames. I did not make significant progress until I developed a tool to show me the impact on thousands of frames at once.
1. Plotting histograms of specific metrics can be very valuable to help filter out false positives in computer vision algorithms.
1. This project would not have been possible without open source tools like ffmpeg, tesseract, opencv, keras, and sklearn that allowed me to focus on the core of the problem.