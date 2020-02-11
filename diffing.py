import matplotlib.pyplot as plt
import pathlib
import cv2
import numpy as np
import base64
import jinja2
import random
import os
import pytesseract #todo


def numpy_img_to_b64_html_src(im):
    ret, buff = cv2.imencode(".bmp", im)
    base64_im = base64.b64encode(buff).decode("utf-8")
    im_src = f"data:image/bmp;base64, {base64_im}"
    return im_src


# Can be made faster??? (less memory) by just writing to file immediately
class HTMLImageDebugger:
    def __init__(self):
        self.images = []

    def add_image(self, image, label="no label"):
        self.images.append({"src": numpy_img_to_b64_html_src(image), "label": label})

    def show(self):
        env = jinja2.Environment(loader=jinja2.PackageLoader("diffing", "templates"))
        template = env.get_template("im_display_template.html")

        path = pathlib.Path("./outputs/html_image_debugger.html")
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(template.render(images=self.images))

        os.system(f"\"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe\" {str(path)}")

dbg = HTMLImageDebugger()


def show_rgb_im(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()

def show_im(im, title=None):
    if title is not None:
        plt.title(title)
    plt.imshow(im, cmap="gray")
    plt.show()


def crop_roi(image, x,y,w,h):
    return image[y:y+h,x:x+w]



def get_image_top_right_roi(im):
    h, w, cdim = im.shape
    return im[:int(.25*h),int(.75*w):int(.85*w)]

def open_frame(streamdir, framenum):
    frames_dir = streamdir / pathlib.Path("frames")
    im_name = pathlib.Path(f"frame{framenum:04}.bmp")

    path_str = str(frames_dir / im_name)
    read_data = cv2.imread(path_str)


    return read_data

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


def get_mean_roi_bgr(dir, framenum):
    return mean_frame(dir, framenum, 1)


def get_skull_contour_from_roi_gs(roi_gs, roi_bgr):
    ret, roi_th1 = cv2.threshold(roi_gs, max(170, np.max(roi_gs) - 25), 255, cv2.THRESH_BINARY)
    med = cv2.medianBlur(roi_th1, 7)
    im, contours, hi = cv2.findContours(med, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None


    biggest = max(contours, key=cv2.contourArea)
    biggest_area = cv2.contourArea(biggest)
    if 69 <= biggest_area <= 131:
        return biggest


    return None


def get_kills_roi_from_skull_contour(roi_gs, skull_contour, roi_bgr):
    x,y,w,h = cv2.boundingRect(skull_contour)

    # If this becomes problematic, use portions of the skull contour
    # and average them across a random sample of the frames
    kills_width = 18
    kills_height = 20

    kills_x = int(x + 18)
    kills_y = int(y)

    last_pix_x = kills_x + kills_width - 1
    last_pix_y = kills_y + kills_height - 1


    big_h, big_w = roi_gs.shape
    max_last_pix_x = big_w - 1
    max_last_pix_y = big_h - 1

    if last_pix_x > max_last_pix_x or last_pix_y > max_last_pix_y:
        return None

    # cv2.rectangle(roi_bgr, (kills_x,kills_y), (kills_x+kills_width,kills_y+kills_height), [0, 255, 0], 2)
    # dbg.add_image(roi_bgr)
    return roi_gs[kills_y:kills_y+kills_height,kills_x:kills_x+kills_width]


def get_binarized_kills_roi(kills_roi):
    ret, binarized = cv2.threshold(kills_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    on_pixels = (binarized == 0).sum()
    prop = on_pixels / binarized.size

    if not (.069 <= prop <= .26):
        return None


    return binarized

def get_connected_comm_roi(conn_comm, index, image):
    ys, xs = np.where(conn_comm == index)
    x_0 = np.min(xs)
    x_1 = np.max(xs)

    y_0 = np.min(ys)
    y_1 = np.max(ys)

    return image[y_0:y_1 + 1, x_0:x_1 + 1]


def get_chars_from_binarized_kills_roi(binarized_kills_roi):
    labels, conn_comm = cv2.connectedComponents(255 - binarized_kills_roi)

    # sort based on position in image, so the digits are not out of order
    left_to_right_labels = list(range(1, labels))
    left_to_right_labels.sort(key=lambda x: np.where(conn_comm == x)[1].min())
    cncs = [get_connected_comm_roi(conn_comm, i, binarized_kills_roi) for i in left_to_right_labels]

    chars_to_ret = []
    for char in cncs:
        on_px = (char == 0).sum()
        if on_px <= 22:
            continue
        h,w = char.shape

        if w > h:
            continue

        chars_to_ret.append(char)

    return chars_to_ret

def center_roi_on_background(roi, bw, bh):
    base = np.full((bh, bw), 255, dtype=np.uint8)
    roi_h, roi_w = roi.shape

    assert bw >= roi_w
    assert bh >= roi_h

    left_x = (bw - roi_w) // 2
    left_y = (bh - roi_h) // 2
    base[left_y:left_y+roi_h, left_x:left_x+roi_w] = roi

    return base

def get_digits_for_frame(dir, frame_num):
    roi_bgr = get_mean_roi_bgr(dir, frame_num)
    roi_gs = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    skull_contour = get_skull_contour_from_roi_gs(roi_gs, roi_bgr)

    if skull_contour is None:
        return None

    # cv2.drawContours(roi_bgr, [skull_contour], -1, [0, 255, 0], -1)
    kills_roi = get_kills_roi_from_skull_contour(roi_gs, skull_contour, roi_bgr)

    if kills_roi is None:
        return None

    binarized_kills_roi = get_binarized_kills_roi(kills_roi)
    if binarized_kills_roi is None:
        return None

    # dbg.add_image(binarized_kills_roi)

    chars = get_chars_from_binarized_kills_roi(binarized_kills_roi)

    if len(chars) == 0:
        return None

    return chars

    #for j, c in enumerate(chars):
    #    dbg.add_image(center_roi_on_background(c, 25, 25), label=f"{frame_num}-{j}")



def get_digits_for_frame_on_bg(dir, frame_num):
    digits = get_digits_for_frame(dir, frame_num) or []
    on_bg = [center_roi_on_background(c, 25, 25) for c in digits]

    return on_bg




def main():
    random.seed(2)
    dir = pathlib.Path(r".\data\ts\mendo\19-05-23--15-44-54")
    xs = []
    for i in range(200):
        x = random.randint(366, 5947)
        digits = get_digits_for_frame(dir, x)
        if digits is None:
            continue


        for d in digits:
            centered = center_roi_on_background(d, 25, 25)
            s = pytesseract.image_to_string(centered, config="--psm 10")
            dbg.add_image(centered, s)

    dbg.show()
    plt.hist(xs, bins=40)
    plt.show()


if __name__ == "__main__":
    main()
