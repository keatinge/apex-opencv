import diffing
import pathlib
import json
import numpy as np
import collections
import sklearn.svm
import random
import label_server
import jinja2
import os
import sklearn.neural_network
import sklearn.tree
import sys
from sklearn import tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.metrics as metrics
from tensorflow.keras import optimizers
import itertools
import matplotlib.pyplot as plt



class GroupHtmlImageDebugger:
    def __init__(self):
        self.groups = []

    def add_image_group(self, ims, label="no label", title="no title"):
        self.groups.append({"label": label, "imgs" : ims, "title" : title})

    def show(self):
        env = jinja2.Environment(loader=jinja2.PackageLoader("ml", "templates"))
        template = env.get_template("im_display_template2.html")

        path = pathlib.Path("./outputs/html_image_debugger3.html")
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(template.render(groups=self.groups, to_b64_func=diffing.numpy_img_to_b64_html_src))

        # TODO: don't use hardcoded chrome
        os.system(f"\"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe\" {path.absolute()}")


def generate_full_dataset(stream_dir):
    labels_path = stream_dir / pathlib.Path("train/labels.json")

    with open(labels_path, "r") as f:
        data = json.load(f)

    debug = diffing.HTMLImageDebugger()

    max_kills = min([l["label"] for l in data["labels"]])
    print(len(data["labels"]))

    for frame_dict in data["labels"]:
        frame_number = frame_dict["frame_num"]
        print(frame_number)
        char_rois = diffing.get_digits_for_frame_on_bg(stream_dir, frame_number)
        frame_dict["chars"] = [char_roi.tolist() for char_roi in char_rois]


    new_file = str(stream_dir / pathlib.Path("train/labels_with_im.json"))
    with open(new_file, "w") as f:
        json.dump(data, f, indent=4)

    #debug.show()



def load_dataset(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)

    for data_dict in data["labels"]:
        data_dict["chars"] = [np.array(c) for c in data_dict["chars"]]


    return data


def load_dataset_as_char_list(dataset_path):
    Char = collections.namedtuple("Char", ["frame_num", "label", "np_arr", "dataset"])
    dataset = load_dataset(dataset_path)

    dataset_name = dataset_path.parts[-3]

    all_chars = []
    for frame_data in dataset["labels"]:
        true_label = frame_data["label"]
        str_label = str(true_label)

        if len(str_label) != len(frame_data["chars"]):
            continue # Wrong number of characters for this frame, data could be wrong

        for np_char, label_str in zip(frame_data["chars"], str_label):
            flat = np_char.reshape((-1,))
            this_char = Char(frame_num=frame_data["frame_num"], label=int(label_str), np_arr=np_char, dataset=dataset_name)
            all_chars.append(this_char)

    return all_chars

def draw_all_chars(list_of_chars):
    debug = diffing.HTMLImageDebugger()
    for this_char in list_of_chars:
        debug.add_image(this_char.np_arr, str(this_char.label))
    debug.show()
def draw_pred_epected(predicted, expected, images):
    debug = diffing.HTMLImageDebugger()

    assert(len(predicted) == len(expected) == len(images))
    cor = 0
    incor = 0
    for p, e, chr_obj in zip(predicted, expected, images):
        if p == e:
            cor += 1
        else:
            debug.add_image(chr_obj.np_arr, label=f"predicted={p}, correct={e}, num={chr_obj.frame_num}")

    prop_cor = cor / len(expected)
    print(f"{cor}/{len(expected)} Correct = {prop_cor:.4f}")
    debug.show()


def classify_frame(stream_dir, frame_num, model):
    char_rois = diffing.get_digits_for_frame_on_bg(stream_dir, frame_num)
    if char_rois is None or len(char_rois) == 0:
        return None, None

    char_rois_for_xs = np.array(char_rois).reshape(-1, 25, 25, 1) #for keras
    predicted = one_hot_to_categorical(model.predict(char_rois_for_xs), list(range(10)))
    as_str = "".join(map(str, predicted))
    return int(as_str), char_rois

def get_all_training_datasets():
    all = []
    mendo_folder = pathlib.Path("./data/ts/mendo/")
    for stream_dir in mendo_folder.iterdir():
        full_label_file = stream_dir / "train" / "labels_with_im.json"
        if full_label_file.is_file():
            all.append(full_label_file)
    return all


def one_hot_to_categorical(vec_of_one_hot, labels):
    indicies = np.argmax(vec_of_one_hot, axis=1)
    return np.array(labels)[indicies]

def grid_search(params):

    params_list = params.items()

    search_spaces = [v for k,v in params_list]
    param_names = [k for k,v in params_list]

    all_choices = list(itertools.product(*search_spaces))

    best = None
    best_errors = None

    results = []
    for choice in all_choices:
        chosen_values = list(choice)
        choices_dict = dict(zip(param_names, chosen_values))


        errors = train_keras(choices_dict)

        results.append((choices_dict, errors))

        print(choices_dict, errors)

        if best is None or errors < best_errors:
            best = choices_dict
            best_errors = errors


    for k, err in results:
        print(err, k)
    print("BEST", best, "ERRORS", best_errors)



def get_chars_train_test_split(lib="keras", train_size=0.5):
    training_datasets = get_all_training_datasets()

    all_chars_list = []
    for dataset_path in training_datasets:
        char_list = load_dataset_as_char_list(dataset_path)
        all_chars_list.extend(char_list)
    print(len(all_chars_list))
    random.shuffle(all_chars_list)

    xs = []
    ys = []
    for char in all_chars_list:
        xs.append(char.np_arr//255)
        ys.append(char.label)

    if lib == "keras":
        xs_lib = np.array(xs).reshape(-1,25,25,1)
        ys_lib = to_categorical(np.array(ys))
    elif lib == "sklearn":
        xs_lib = np.array(xs).reshape(-1,25*25)
        ys_lib = np.array(ys)
    else:
        raise ValueError("lib must be keras or sklearn")

    x_train, x_test, y_train, y_test, ch_train, ch_test = sklearn.model_selection.train_test_split(xs_lib, ys_lib, all_chars_list, train_size=train_size)
    return x_train, x_test, y_train, y_test, ch_train, ch_test

def evaluate_model(ch_test, expected, predicted, name="no name"):
    print("Evaluating model", name)
    print(sklearn.metrics.classification_report(expected, predicted, digits=5))
    conf = sklearn.metrics.confusion_matrix(expected, predicted)
    print(conf)
    gdb = GroupHtmlImageDebugger()
    count = 0
    for char, exp, pred in zip(ch_test, expected, predicted):
        # if exp != pred:
        gdb.add_image_group([char.np_arr], label=f"{pred} actual={exp}", title=f"{char.frame_num} from {char.dataset}")
            # count += 1
    gdb.show()
    print("Test size", expected.shape)

    return sklearn.metrics.classification_report(expected, predicted, digits=5, output_dict=True)["accuracy"]






def train_knn():
    x_train, x_test, y_train, y_test, ch_train, ch_test = get_chars_train_test_split(lib="sklearn")
    parameters_knn = {}
    model = sklearn.model_selection.GridSearchCV(
        estimator=sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, algorithm="brute", p=1),
        param_grid=parameters_knn, verbose=10)

    history = model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    expected = y_test
    evaluate_model(ch_test, expected, predicted, "sk-knn")







def train_keras():
    x_train, x_test, y_train, y_test, ch_train, ch_test = get_chars_train_test_split(lib="keras", train_size=0.5)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(25,25,1), data_format="channels_last"))
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

    plt.title("Training Loss")
    plt.ylabel("Categorical Cross-Entropy")
    plt.xlabel("Epochs")
    plt.plot(history.history["loss"])
    plt.show()

    one_hot_pred = model.predict(x_test)
    predic_cat = one_hot_to_categorical(one_hot_pred, list(range(10)))
    expect_cat = one_hot_to_categorical(y_test, list(range(10)))

    acc = evaluate_model(ch_test, expect_cat, predic_cat, "keras")
    #model.save(f"64cnn_100d_{10_000*acc:.0f}.h5")

    return model


def full_classify():
    model = train_keras()
    testing_dir = pathlib.Path(r".\data\ts\mendo\19-05-23--18-42-01")
    debug = GroupHtmlImageDebugger()
    ks = []
    xs = []

    debug.show()

    for frame in testing_dir.glob("frames/*.bmp"):
        print(frame)
        frame_num = label_server.get_frame_number_from_filename(frame.name)
        kills, chr_ims = classify_frame(testing_dir, frame_num, model)

        if kills is None:
            continue
        xs.append(frame_num)
        ks.append(kills)

        frame = diffing.open_frame(testing_dir, frame_num)
        print(frame.shape)
        plt.imshow(frame)
        debug.add_image_group([frame[0:100,1200:]], label=f"{kills}", title=f"{frame_num}")
    debug.show()


if __name__ == "__main__":
    full_classify()

