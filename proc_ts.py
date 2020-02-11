import os
import re
import cv2
import time
import json
import pathlib
import datetime
import twitch_dl
import subprocess
import collections
import pytesseract
import dateutil.parser

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_FPS = .33
SAMPLE_SPF = 1/SAMPLE_FPS

def get_num_from_ts_filename(ts_filename):
    num = int(re.search(r"^(\d+)", ts_filename).group(1))
    return num


def collapse_existing_ts_files(stream_dir):
    print("Collapsing TS files @", stream_dir)
    ts_files = [file for file in os.listdir(stream_dir) if file.endswith("ts")]
    ts_files.sort(key=get_num_from_ts_filename)


    if len(ts_files) <= 0:
        raise ValueError("There are no TS files to collapse!")

    if len(ts_files) == 1:
        return ts_files[0]

    with open(os.path.join(stream_dir, ts_files[0]), "ab") as ts_dest_f:
        for ts_src_path in ts_files[1:]:
            full_src_path = os.path.join(stream_dir, ts_src_path)
            with open(full_src_path, "rb") as ts_src_f:
                ts_dest_f.write(ts_src_f.read())
            os.remove(full_src_path)

    if "muted" in ts_files[0]:
        new_name = f"{get_num_from_ts_filename(ts_files[0])}.ts"
        os.rename(ts_files[0], new_name)
        ts_files[0] = new_name

    print(f"Collapsed {len(ts_files)} ts files into 1")

    return ts_files[0]


def sample_frames(src_video_file, output_loc):
    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg", "-i", src_video_file, "-vf", f"fps={SAMPLE_FPS},crop=in_w:0.4*in_h:0:0", "-nostdin", os.path.join(output_loc, "frame%04d.bmp")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = ffmpeg_proc.communicate()
    print(out)
    print(err)


def nuke_folder(folder_path):
    print("Nuking folder @", folder_path)
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))

def generate_frames(stream_dir, src_file):
    frames_dir = os.path.join(stream_dir, "frames")
    print("Generating frames to", frames_dir)
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)
    else:
        pass # TODO TODO TODO
        #nuke_folder(frames_dir)
    sample_frames(os.path.join(stream_dir, src_file), frames_dir)
    return frames_dir

def get_frame_markers(full_im_path):
    im = Image.open(full_im_path)
    s = pytesseract.image_to_string(im)
    lower_text = s.lower()

    markers = {
        "is_game_start": "select legend" in lower_text,
        "is_game_end": ("placed" in lower_text and "of" in lower_text),
    }

    return markers

def get_frame_number_from_frame_file(frame_file):
    return int(re.search(r"frame(\d*)\.bmp", frame_file).group(1))

def find_game_start_ends(stream_dir):
    print("Calculating game starts and ends from", stream_dir)
    im_files = os.listdir(stream_dir)
    im_files.sort(key=get_frame_number_from_frame_file)

    gs_found_frames = []
    ge_found_frames = []


    for i,im_file in enumerate(im_files[:-1]):
        if i % 10 == 0:
            print("{:02d} {:.2f}% complete".format(i, 100* i/len(im_files)))

        full_im_file_path = os.path.join(stream_dir, im_file)
        next_full_im_file_path = os.path.join(stream_dir, im_files[i + 1])

        markers = get_frame_markers(full_im_file_path)
        #print(i, markers)
        if markers["is_game_start"] and not get_frame_markers(next_full_im_file_path)["is_game_start"]:
            print("FOUND GS---------------", im_file)
            gs_found_frames.append(im_file)

        if markers["is_game_end"] and not get_frame_markers(next_full_im_file_path)["is_game_end"]:
            print("FOUND GE---------------", im_file)
            ge_found_frames.append(im_file)


    print("Starts", gs_found_frames)
    print("Ends", ge_found_frames)



    return {"starts" : gs_found_frames, "ends" : ge_found_frames}

def first_frame_after(frame, others):

    ref_frame_num = get_frame_number_from_frame_file(frame)
    after = [f for f in others if get_frame_number_from_frame_file(f) > ref_frame_num]

    if len(after) == 0:
        return None

    first = min(after, key=get_frame_number_from_frame_file)
    return first

def parse_game_starts_ends(starts, ends):
    wins = []
    for i, gs in enumerate(starts):
        next_gs = first_frame_after(gs, starts[i+1:])
        next_ge = first_frame_after(gs, ends)

        # Careful - all of this will break for a frame0000.bmp, which ffmpeg thankfully
        # doesn't produce, so this should all be fine


        if next_gs:
            next_gs_frame_num = get_frame_number_from_frame_file(next_gs)

        if next_ge:
            next_ge_frame_num = get_frame_number_from_frame_file(next_ge)


        if next_ge and next_gs and next_ge_frame_num < next_gs_frame_num:
            wins.append({"start_frame_file" : gs, "end_frame_file" : next_ge})
            print("Good start and end", gs, next_ge)

        if next_ge and next_gs and next_ge_frame_num >= next_gs_frame_num:
            print("Bad game (not win), never found ending from", gs, next_gs)

        if next_ge and not next_gs:
            wins.append({"start_frame_file" : gs, "end_frame_file" : next_ge})
            print("Good start and end", gs, next_ge)

        if not next_ge and next_gs:
            print("This is a broken game with bad time bound", gs, next_gs)

        if not next_ge and not next_gs:
            print("Game still in progress")

    return wins

def frame_num_to_time_seconds(frame_num):
    return frame_num * SAMPLE_SPF

def ffmpeg_clip_video(input_file, start_t, end_t, output):


    if end_t is None:
        args = ["ffmpeg", "-ss", str(start_t), "-i", input_file, "-codec", "copy", "-nostdin", output]
    else:
        duration = end_t - start_t
        args = ["ffmpeg", "-ss", str(start_t), "-i", input_file, "-codec", "copy", "-t", str(duration), "-nostdin", output]

    print("Running", " ".join(args))
    ffmpeg_proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = ffmpeg_proc.communicate()
    print(out, err)


def ffmpeg_get_duration(input_file):
    ffmpeg_proc = subprocess.Popen(["ffmpeg", "-i", input_file, "-nostdin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = ffmpeg_proc.communicate()
    h,m,s = map(int, re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})", err.decode('utf-8')).groups())
    return h,m,s

def get_frame_file_from_num(num):
    return f"frame{num:04d}.bmp"

def get_mean_section(stream_dir, frame_num, jump_ahead):
    sum_image = None
    count = 0

    frames_folder = os.path.join(stream_dir, "frames")

    for i in range(frame_num-jump_ahead, frame_num+jump_ahead+1):
        frame_filename = get_frame_file_from_num(i)

        frame_full_file_path = os.path.join(frames_folder, frame_filename)
        frame = cv2.imread(frame_full_file_path)

        if frame is None:
            continue

        count += 1

        if sum_image is None:
            sum_image = frame.astype(np.float)
        else:
            sum_image += frame

    frame_im = (sum_image/float(count)).astype(np.uint8)
    h, w, channels = frame_im.shape
    #section = frame_im[:int(.1 * h), int(.75 * w):int(.9 * w), :]

    # Using exact same image size, but now ffmpeg is pre cropping to .4h, and (.25)(.4)h = (.1)h
    section = frame_im[:int(.25 * h), int(.75 * w):int(.9 * w), :]

    return section

def get_connected_comm_roi(conn_comm, index, image):
    ys, xs = np.where(conn_comm == index)
    x_0 = np.min(xs)
    x_1 = np.max(xs)

    y_0 = np.min(ys)
    y_1 = np.max(ys)

    return image[y_0:y_1+1,x_0:x_1+1]


def make_frame_debug_dir_if_not_exist(stream_dir, frame_file):
    debug_path = os.path.join(stream_dir, "debug")
    debug_frame_path = os.path.join(debug_path, str(get_frame_number_from_frame_file(frame_file)))

    #print("***",debug_frame_path)
    if not os.path.isdir(debug_frame_path):
        os.mkdir(debug_frame_path)
    return debug_frame_path



def debug_save_json(stream_dir, frame_file, data_dict, name):
    frame_dir = make_frame_debug_dir_if_not_exist(stream_dir, frame_file)
    frame_num = get_frame_number_from_frame_file(frame_file)
    file_name = f"{frame_num}_{name}.txt"

    with open(os.path.join(frame_dir, file_name), "w") as f:
        f.write(json.dumps(data_dict, indent=4))



def debug_save_image(stream_dir, frame_file, roi_th, name):
    frame_dir = make_frame_debug_dir_if_not_exist(stream_dir, frame_file)
    frame_num = get_frame_number_from_frame_file(frame_file)
    file_name = f"{frame_num}_{name}.bmp"
    cv2.imwrite(os.path.join(frame_dir, file_name), roi_th)
    pass


def blend_gs_onto_rgb(rgb_image, gs_overlay):
    # rgb_gs = np.stack((np.zeros(gs_overlay.shape), 255 * 1-gs_overlay ,np.zeros(gs_overlay.shape)), axis=-1)
    rgb_image_cp = np.copy(rgb_image)
    rgb_image_cp[gs_overlay == 0, 1] = 255
    rgb_image_cp[gs_overlay == 0, 0] = 0
    rgb_image_cp[gs_overlay == 0, 2] = 0
    return rgb_image_cp


def get_kills_roi_for_frame_num(stream_dir, frame_file):
    debug = False
    # print(frame_file)
    frame_num = get_frame_number_from_frame_file(frame_file)

    section = get_mean_section(stream_dir, frame_num, 1)

    if debug:
        debug_save_image(stream_dir, frame_file, section, "mean_section")

    frame_gs = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    ret, th_status_sec = cv2.threshold(frame_gs, max(150, np.max(frame_gs)-20), 255, cv2.THRESH_BINARY_INV)


    med_blur = cv2.medianBlur(th_status_sec, 5)
    if debug:
        debug_save_image(stream_dir, frame_file, blend_gs_onto_rgb(section, med_blur), "med_blur_status_sect")
    im2, contours, hi = cv2.findContours(med_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # debug_save_contours(stream_dir, frame_file, contours)
    # cv2.drawContours(section, contours, -1, [0, 255, 0], 1)

    if len(contours) == 0:
        print("No contours")
        return section, None
    actual_hi = hi[0]
    parent_candidates = [c for c in actual_hi if c[3] == -1]

    all_children = []

    if len(parent_candidates) == 1:
        child_index = parent_candidates[0][2]
        while child_index != -1:
            all_children.append(contours[child_index])
            child_index = actual_hi[child_index][0]
    else:

        all_children = contours[0]

    viable_children = [c for c in all_children if cv2.contourArea(c) >= 30]

    if len(viable_children) == 1:
        best_cand = viable_children[0]
        best_cand = max(viable_children, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best_cand)

        x1_roi = x + int(1.22 * w)
        y1_roi = y

        #x2_roi = x + int(2.45 * w)
        x2_roi = x + int(2.5*w)
        y2_roi = y + int(1.8 * h)


        if debug:
            cv2.rectangle(section, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
            debug_save_image(stream_dir, frame_file, section, "kills_bounding_rect")

        roi = frame_gs[y1_roi:y2_roi, x1_roi:x2_roi]
        if roi.size == 0:
            print("Super messed up roi, out of bounds")
            return section, None


        # actually should probably average here instead of threshing
        roi_ret, roi_th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if debug:
            debug_save_image(stream_dir, frame_file, roi_th, "kills_bounding_rect_th")

        # Remove components that are too small
        labels, conn_comm = cv2.connectedComponents(255-roi_th)
        total_area = np.prod(roi_th.shape)
        removed_indicies = []
        for i in range(1, labels):
            count = np.sum(conn_comm == i)
            area = count / total_area

            if area <= .05:
                roi_th[conn_comm == i] = 255
                removed_indicies.append(i)


        # Separate individual characters character
        final_cnn = [i for i in range(1, labels) if i not in removed_indicies]

        # Sort by their smallest x coordinate, so we always read left to right
        final_cnn.sort(key=lambda x: np.min(np.where(conn_comm == x)[1]))

        if len(final_cnn) == 1 or len(final_cnn) == 2:
            final_rois = []
            for this_cnn in final_cnn:
                ch_roi = get_connected_comm_roi(conn_comm, this_cnn, roi_th)
                final_rois.append(ch_roi)
            return roi_th, final_rois
        else:
            print("Too many characters detected")
            return roi_th, None

    else:
        print("Viable children", len(viable_children))
        return section, None


def center_roi(roi):
    onY, onX = np.where(roi == 0)
    x1, x2 = min(onX), max(onX)
    y1, y2 = min(onY), max(onY)

    just_char_im = roi[y1:y2+1, x1:x2+1]


    chr_height, chr_width = just_char_im.shape
    big_height, big_width = 25, 25
    background_im = np.full((big_height, big_width), 255)

    overlay_h, overlay_w = just_char_im.shape


    p_x = (big_width - overlay_w) // 2
    p_y = (big_height - overlay_h) // 2

    background_im[p_y:p_y+overlay_h,p_x:p_x+overlay_w] = just_char_im

    return background_im


def get_kills_on_frame(stream_dir, frame_file):
    # frame_path = get_frame_full_path(frame_num)
    #print(frame_path)


    full_roi, all_rois = get_kills_roi_for_frame_num(stream_dir, frame_file)

    if all_rois is None:
        # plt.title("FAIL")
        # plt.imshow(full_roi)
        # plt.show()

        return None

    full_str = ""
    strs = []

    desired_width = 40


    for i,roi in enumerate(all_rois):

        centered = center_roi(roi)

        # plt.imshow(centered)
        # plt.show()
        debug_save_image(stream_dir, frame_file, centered, f"centered{i}")
        centered_resized = roi
        # h,w = roi.shape
        # k = desired_width/w

        # Hard to say if resizing is really better or not
        # resized = cv2.resize(roi, (int(w*k), int(h*k)))
        # resized[resized != 255] = 0

        # debug_save_image(stream_dir, frame_file, im, f"resized{i}")


        im = Image.fromarray(centered)
        kills_str = pytesseract.image_to_string(im, config="--psm 10")
        strs.append(kills_str)

    # print(full_str)

    common_mistakes = {
        "i" : "1",
        "l": "1",
        "s" : "5",
        "a" : "0",
        "o" : "0",
        "]" : "1",
        "[" : "1",
    }
    new_strs = [(common_mistakes[s] if s in common_mistakes else s) for s in strs]

    full_str = "".join(new_strs)
    only_dig = [d for d in full_str if d.isdigit()]

    debug_data = {
        "strs" : strs,
        "strs_after_mistakes_filter" : new_strs,
        "only_dig" : only_dig,
    }

    debug_save_json(stream_dir, frame_file, debug_data, "tesseract_debug")
    if len(only_dig) == 0:
        return None

    num = int("".join(only_dig))

    if num >= 40 or num == 0:
        return None

    return num


def get_kills_for_game(stream_dir, end_frame_num):
    curr_frame_num = end_frame_num
    while get_kills_on_frame(stream_dir, get_frame_file_from_num(curr_frame_num)) is None:
        curr_frame_num -= 1

    last_gameplay_frame = curr_frame_num
    #n_frames_to_check = 60 TODO TODO UNCOMMENT

    n_frames_to_check = 120

    kills_series = []
    i_series = []
    skipped_frames = []
    non_skipped = []
    frame_kill_list = []

    #for i in range(last_gameplay_frame-n_frames_to_check+1, last_gameplay_frame+1):
    for i in range(last_gameplay_frame, last_gameplay_frame-n_frames_to_check-1, -1):
        kills = get_kills_on_frame(stream_dir, get_frame_file_from_num(i))
        if kills is not None:
            kills_series.append(kills)
            i_series.append(i)
            non_skipped.append(i)
            frame_kill_list.append((i, kills))
        else:
            skipped_frames.append(i)

        if len(kills_series) == 20:
            break


    print(kills_series)
    print("Skipped frames", skipped_frames)
    print("Non skipped", non_skipped)
    print("Framekill", frame_kill_list)

    total_frames = len(skipped_frames) + len(non_skipped)
    print("Skipped", len(skipped_frames), "of", total_frames, "or {0:.2%}".format(len(skipped_frames)/total_frames))

    counter = collections.Counter(kills_series)

    # If all the kills are the same, it's almost certainly correct
    if len(counter) == 1:
        return kills_series[-1]

    max_k = max(kills_series)
    max_k_occ = counter[max_k]

    # If the last el is the max and it occurs atleast 3 times it's probably correct
    if kills_series[-1] == max_k and max_k_occ >= 3:
        return kills_series[-1]

    # Otherwise the median will be a decent approximation
    return np.median(kills_series)


    # Debugging info
    # plt.title("Kills for")
    # plt.plot(i_series, kills_series)
    # plt.show()
    # print("KS", kills_series)
    # print("IS", i_series)

def get_frame_num_for_first_kill(stream_dir, start_frame_num, end_frame_num):

    for frame_num in range(start_frame_num, end_frame_num):
        kills = get_kills_on_frame(stream_dir, get_frame_file_from_num(frame_num))
        if kills is not None and kills <= 5:
            return frame_num
        if kills is not None:
            break


    return start_frame_num


def make_video_from_game_data(stream_dir, game_data):
    output_folder_full_path = os.path.join(stream_dir, "fin")
    if not os.path.exists(output_folder_full_path):
        os.mkdir(output_folder_full_path)


    file_name = "{:.1f}-{:.1f} {} kills {:.1f} mins.ts".format(game_data["start_sec"], game_data["end_sec"], game_data["estimated_kills"],
                                                               game_data["duration_minutes"])  # these could actually collide, pretty unlikely but possible -- should fix
    full_output_filepath = os.path.join(output_folder_full_path, file_name)

    ffmpeg_clip_video(input_file=game_data["input_file"],
                      start_t=game_data["start_sec"],
                      end_t=game_data["end_sec"],
                      output=full_output_filepath)

    return file_name


def create_video_data(stream_dir, ts_file, win):

    debug_dir = os.path.join(stream_dir, "debug")
    if not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)


    print("Creating video for", win, "from", ts_file)
    big_ts_file_full_path = os.path.join(stream_dir, ts_file)
    start_frame = win["start_frame_file"]
    end_frame = win["end_frame_file"]

    start_frame_num = get_frame_number_from_frame_file(start_frame)
    end_frame_num = get_frame_number_from_frame_file(end_frame)


    first_kill_frame_num = get_frame_num_for_first_kill(stream_dir, start_frame_num, end_frame_num)
    desired_dt_sec = 60

    adjusted_first_frame_num = max(start_frame_num, int(first_kill_frame_num-(SAMPLE_FPS*desired_dt_sec)))

    start_sec = frame_num_to_time_seconds(adjusted_first_frame_num)
    end_sec = frame_num_to_time_seconds(end_frame_num)



    est_kills_in_game = get_kills_for_game(stream_dir, end_frame_num)
    duration_minutes = (1+end_sec-start_sec)/60


    game_data = {
        "input_file" : big_ts_file_full_path,
        "estimated_kills" : est_kills_in_game,
        "duration_minutes" : duration_minutes,
        "start_sec" : start_sec,
        "end_sec" : end_sec,
        "start_frame": adjusted_first_frame_num,
        "end_frame" : end_frame_num
    }

    return game_data


def overlay_on_fixed_image(to_overlay, big_w, big_h):
    overlay_h, overlay_w = to_overlay.shape

    if overlay_h > big_h or overlay_w > big_w:
        return None

    fixed_image = np.full((big_h, big_w), 255, dtype=np.uint8)
    top_left_x = (big_w - overlay_w) // 2
    top_left_y = (big_h - overlay_h) // 2

    fixed_image[top_left_y:top_left_y+overlay_h,top_left_x:top_left_x+overlay_w] = to_overlay
    return fixed_image



def generate_train_folder(stream_dir):
    frames_dir = pathlib.Path(stream_dir) / pathlib.Path("frames")
    train_dir = pathlib.Path(stream_dir) / pathlib.Path("train")

    train_dir.mkdir(exist_ok=True)

    for i,file in enumerate(frames_dir.iterdir()):

        #file = frames_dir / pathlib.Path("frame5766.bmp")
        print(file)
        roi_th, chars_rois = get_kills_roi_for_frame_num(stream_dir, file.name)
        frame_number = get_frame_number_from_frame_file(file.name)

        if chars_rois is not None:
            for ch_i, chr_im in enumerate(chars_rois):
                overlay_im = overlay_on_fixed_image(chr_im, 25, 25)
                if overlay_im is None:
                    continue
                file_name = "frame{:04}-ch{}.bmp".format(frame_number, ch_i)
                full_path = train_dir / pathlib.Path(file_name)
                cv2.imwrite(str(full_path), overlay_im)










def cleanup_all_after_win(stream_dir, big_ts_file, last_win):
    last_win_end_frame_file = last_win["end_frame_file"]
    frame_num = get_frame_number_from_frame_file(last_win_end_frame_file)
    frame_time = frame_num_to_time_seconds(frame_num)

    ts_file_start_time = get_num_from_ts_filename(big_ts_file)
    delta_time = datetime.timedelta(seconds=frame_time)
    new_ts_start_datetime = ts_file_start_time + delta_time
    new_ts_filename = twitch_dl.create_ts_filename_from_date(new_ts_start_datetime)

    BUFFER_TIME_SEC = 10

    old_ts_full_path = os.path.join(stream_dir, big_ts_file)
    new_ts_full_path = os.path.join(stream_dir, new_ts_filename)
    print("Creating new big ts file: ", new_ts_full_path)
    ffmpeg_clip_video(
        input_file=old_ts_full_path,
        start_t=frame_time-BUFFER_TIME_SEC,
        end_t=None,
        output=new_ts_full_path
    )

    print("Removing", old_ts_full_path)
    #os.remove(old_ts_full_path) TODO TODO TODO TODO UNCOMMENT THIS





def process_directory(stream_dir):

    t0 = time.time()
    # RUN THE WHOLE THING
    big_ts_file = collapse_existing_ts_files(stream_dir)
    frames_dir = generate_frames(stream_dir, big_ts_file) # TODO UNCOMMENT

    frames_dir = os.path.join(stream_dir, "frames")
    starts_ends = find_game_start_ends(frames_dir)
    wins = parse_game_starts_ends(starts_ends["starts"], starts_ends["ends"])

    # RUN JUST A SPECIFIC VIDEO
    # big_ts_file = collapse_existing_ts_files(stream_dir)
    # frames_dir = os.path.join(stream_dir, "frames")
    #
    # starts_ends = {
    #     "starts" : [
    #         'frame0051.bmp', 'frame0115.bmp', 'frame0436.bmp', 'frame0766.bmp', 'frame1098.bmp', 'frame1451.bmp', 'frame1628.bmp', 'frame1838.bmp', 'frame2194.bmp', 'frame2386.bmp', 'frame2524.bmp', 'frame2839.bmp', 'frame2976.bmp', 'frame3229.bmp', 'frame3382.bmp', 'frame3778.bmp', 'frame4117.bmp', 'frame4430.bmp', 'frame4481.bmp'],
    #     "ends" : [
    #         'frame0041.bmp', 'frame0425.bmp', 'frame0756.bmp', 'frame1065.bmp', 'frame1433.bmp', 'frame1435.bmp', 'frame1617.bmp', 'frame1814.bmp', 'frame1974.bmp', 'frame2830.bmp', 'frame3210.bmp', 'frame3673.bmp', 'frame4107.bmp', 'frame4864.bmp']
    #
    # }
    # wins = parse_game_starts_ends(starts_ends["starts"], starts_ends["ends"])


    for win in wins:
        game_data = create_video_data(stream_dir, big_ts_file, win)
        video_file = make_video_from_game_data(stream_dir, game_data)

        fin_folder = os.path.join(stream_dir, "fin")
        with open(os.path.join(fin_folder, video_file.replace("ts", "json")), "w") as f:
            json.dump(game_data, f, indent=4)



    print("Wins:", wins)

    # if len(wins) != 0:
    #     pass
    #     #cleanup_all_after_win(dir, big_ts_file, wins[-1]) # TODO: Uncomment to delete files when done

    print("Completed in", round(time.time() - t0, 2), "seconds")


    #Started 6:40


if __name__ == "__main__":
    process_directory("./data/ts/mendo/19-05-23--18-42-01")

