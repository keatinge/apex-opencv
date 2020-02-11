import tornado
import tornado.web
import tornado.ioloop
import pathlib
import collections
import itertools
import proc_ts
import base64
import re
import cv2
import diffing
import json


def get_data_dir():
    return pathlib.Path("./data")


def get_saved_userpaths():
    subdir = get_data_dir() / pathlib.Path("ts")
    usernames = [x for x in subdir.iterdir() if x.is_dir()]
    return usernames


def get_streams_with_frames_for_userpath(userpath):
    StreamDir = collections.namedtuple("StreamDir", ["path", "num_frames"])
    framedirs = []
    for stream_dir in userpath.iterdir():
        if not stream_dir.is_dir():
            continue

        frame_dir = stream_dir / pathlib.Path("frames")
        if frame_dir.is_dir():
            num_frames = sum(1 for f in frame_dir.iterdir())
            if num_frames != 0:
                framedirs.append(StreamDir(path=stream_dir, num_frames=num_frames))
    return framedirs


def get_frame_number_from_filename(frame_filename):
    frame_num = re.search(r"frame(\d*).bmp", frame_filename).group(1)
    return int(frame_num)


def file_to_frame_obj(file_path):
    Frame = collections.namedtuple("Frame", ["path", "frame_num"])
    return Frame(path=file_path, frame_num=get_frame_number_from_filename(file_path.name))


def get_frameb64_from_frame(frame):
    FrameB64 = collections.namedtuple("FrameB64", ["frame", "src"])
    im = cv2.imread(str(frame.path))
    roi = diffing.get_image_top_right_roi(im)
    html_src = diffing.numpy_img_to_b64_html_src(roi)
    return FrameB64(frame=frame, src=html_src)


def get_frames_in_path(streampath):
    framesdir = streampath / pathlib.Path("frames")


    all_frames = [file_to_frame_obj(x) for x in framesdir.glob("*.bmp")]
    all_frames.sort(key=lambda x: x.frame_num)

    n = 0
    for frame in all_frames:
        print(n)
        n += 1
        frame_b64_obj = get_frameb64_from_frame(frame)
        yield frame_b64_obj

    print(n)



def get_all_streamdirs_with_frames():
    all_userpaths = get_saved_userpaths()
    stream_dirs_nested = [get_streams_with_frames_for_userpath(userpath) for userpath in all_userpaths]
    all_stream_dirs = list(itertools.chain(*stream_dirs_nested))

    return all_stream_dirs


class IndexHandler(tornado.web.RequestHandler):
    async def get(self):
        self.render("./templates/index.html", stream_dirs=get_all_streamdirs_with_frames())

class LabelHandler(tornado.web.RequestHandler):
    async def get(self):
        path = pathlib.Path(self.get_argument("path"))
        self.render("./templates/label.html", path=path, frame_gen=get_frames_in_path(path))

    async def post(self):
        req_data = json.loads(self.request.body)
        path, labels_dict, time = req_data["path"], req_data["labels"], req_data["time"]

        data_pts = []

        for str_framenum, label in sorted(labels_dict.items(), key=lambda x: int(x[0])):
            data_pts.append({"frame_num" : int(str_framenum), "label" : label})

        full_path = pathlib.Path(".") / pathlib.Path(path) /  pathlib.Path("train/labelsTODO.json")
        with open(str(full_path), "w") as f:
            json.dump({
                "labels": data_pts,
                "path": path,
                "time_8601": time
            }, f, indent=4)
        print("Finished writing to", full_path)

        self.write({"num_labels_written" : len(data_pts)})





if __name__ == "__main__":
    #get_frames_in_path(pathlib.Path(r"data\ts\mendo\19-05-05--19-45-04"))
    app = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/label", LabelHandler),
    ], compiled_template_cache=False)

    app.listen(80)
    tornado.ioloop.IOLoop.current().start()