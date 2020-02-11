import requests
import datetime
import dateutil.parser
import threading
import queue
import time
import pytz
import sys
import re
import os

CHROME = {"User-Agent" : "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}

def get_access_token_and_sig_for_stream(channel_name):
    access_token_url_fmt = "https://api.twitch.tv/api/channels/{channel_name}/access_token?need_https=true&oauth_token=&platform=_&player_backend=mediaplayer&player_type=site"
    access_token_url = access_token_url_fmt.format(channel_name=channel_name)
    full_headers = {**CHROME, "Client-ID": "jzkbprff40iqj646a697cyrvl0zt2m6"}
    json_resp = requests.get(access_token_url, headers=full_headers).json()
    return json_resp["token"], json_resp["sig"]


def get_access_token_and_sig_for_vod(vod_id):
    access_token_url_fmt = "https://api.twitch.tv/api/vods/{vod_id}/access_token?need_https=true&oauth_token=&platform=_&player_backend=mediaplayer&player_type=site"
    access_token_url = access_token_url_fmt.format(vod_id=vod_id)
    full_headers = {**CHROME, "Client-ID": "jzkbprff40iqj646a697cyrvl0zt2m6"}
    json_resp = requests.get(access_token_url, headers=full_headers).json()

    return json_resp["token"], json_resp["sig"]

def find_stream_in_m3u8_listing(m3u8_listing_url, access_token, sig):

    full_params = {"allow_source": True, "token": access_token, "sig": sig}
    m3u8_listing_req = requests.get(m3u8_listing_url, params=full_params)

    if m3u8_listing_req.status_code == 404:
        raise Exception(f"{channel_name} is not streaming")
    else:
        # Only returns the 1080p60fps stream everything else is ignored
        match1080 = re.search(r'NAME="1080p60.*?(https[^\n]*?\.m3u8)', m3u8_listing_req.text, flags=re.DOTALL)
        if match1080:
            return match1080.group(1)

        match900 = re.search(r'NAME="900p60.*?(https[^\n]*?\.m3u8)', m3u8_listing_req.text, flags=re.DOTALL)
        if match900:
            return match900.group(1)

        raise Exception("Couldn't find a 1080p60 or 900p60 stream in:\n" + m3u8_listing_req.text)

def get_stream_m3u8_for_channel(channel_name):
    url = "https://usher.ttvnw.net/api/channel/hls/{channel_name}.m3u8".format(channel_name=channel_name)
    token, sig = get_access_token_and_sig_for_stream(channel_name)

    return find_stream_in_m3u8_listing(url, token, sig)

def get_stream_m3u8_url_for_vod(vod_id):
    url = "https://usher.ttvnw.net/vod/{vod_id}.m3u8".format(vod_id=vod_id)
    token, sig = get_access_token_and_sig_for_vod(vod_id)

    return find_stream_in_m3u8_listing(url, token, sig)

def get_ts_files_from_stream_m3u8(m3u8_file_text):
    m3u8_regex = r"#EXT-X-PROGRAM-DATE-TIME:([\dT:.Z-]*)\n.*\n(https:.*)"

    all = []
    for match in re.finditer(m3u8_regex, m3u8_file_text):
        time, url = match.groups()
        all.append({"time": dateutil.parser.parse(time), "url" : url})

    return all

def get_ts_files_from_vod_m3u8_text(m3u8_index_url, m3u8_file_text):
    all_ts_file_urls = []
    ts_file_base_url = "/".join(m3u8_index_url.split("/")[:-1])
    for line in m3u8_file_text.splitlines():
        if not line.endswith("ts"):
            continue

        full_url = "{}/{}".format(ts_file_base_url, line)
        all_ts_file_urls.append(full_url)

    return all_ts_file_urls

def get_now_date_str():
    return datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")


def get_time_est(dt_obj):
    eastern = pytz.timezone("US/Eastern")
    return dt_obj.astimezone(eastern)



def setup_directory(channel_name):
    streamer_dir_path = f"./data/ts/{channel_name}"
    if not os.path.isdir(streamer_dir_path):
        os.mkdir(streamer_dir_path)


    now_date_str = get_now_date_str()
    this_stream_path = streamer_dir_path + f"/{now_date_str}"

    if not os.path.isdir(this_stream_path):
        os.mkdir(this_stream_path)

    return this_stream_path


def create_ts_filename_from_date(date_to_use):
    return "{}.ts".format(date_to_use.isoformat().replace(":", " "))


def download_and_save_tsf(tsf, output_folder):
    url = tsf["url"]
    file_time = tsf["time"]

    r = requests.get(url, stream=True)
    if r.status_code == 200:
        this_file_name =  create_ts_filename_from_date(file_time)
        with open(os.path.join(output_folder, this_file_name), "wb") as f:
            for chunk in r:
                f.write(chunk)
    else:
        print("GOT BAD STATUS CODE FOR", tsf, file=sys.stderr)
        print("THE REQUEST WAS", r, file=sys.stderr)

def stream_video(m3u8_stream_url, output_folder):
    last_downloaded_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)

    while True:
        m3u8_text = requests.get(m3u8_stream_url).text
        ts_files = sorted(get_ts_files_from_stream_m3u8(m3u8_text), key=lambda x: x["time"])
        new_ts = [tsf for tsf in ts_files if tsf["time"] > last_downloaded_time]
        last_downloaded_time = max([last_downloaded_time] + [ts["time"] for ts in new_ts])


        for tsf in new_ts:
            print("Downloading ts from", get_time_est(tsf["time"]))
            download_and_save_tsf(tsf, output_folder)
        print("LDT is now", get_time_est(last_downloaded_time), "used", len(new_ts), "of", len(ts_files))


        while datetime.datetime.now(datetime.timezone.utc) < last_downloaded_time + datetime.timedelta(seconds=15):
            time.sleep(0.1)
            print("s ", end="")
            sys.stdout.flush()

        print("Sleep end")

        if datetime.datetime.now(datetime.timezone.utc) - last_downloaded_time > datetime.timedelta(minutes=2):
            print("Got nothing new for 2 minutes... ending streaming")


def download_ts_worker(save_directory, ts_url_queue):
    while True:
        try:
            ts_url = ts_url_queue.get(False)
        except queue.Empty:
            print("Worker finished")
            break

        print("Starting", ts_url)
        resp = requests.get(ts_url, headers=CHROME, stream=True)

        name = ts_url.split("/")[-1]  # extract file name, like 15.ts
        full_file_path = os.path.join(save_directory, name)
        with open(full_file_path, "wb") as f:
            for chunk in resp:
                f.write(chunk)

        print("Finished", ts_url)
        ts_url_queue.task_done()


def download_all_ts_files(dir, ts_urls):
    q = queue.Queue()
    for url in ts_urls:
        q.put(url)

    num_threads = 3
    for i in range(num_threads):
        t = threading.Thread(target=lambda: download_ts_worker(dir, q))
        t.start()
    q.join()



def download_and_save_stream(channel_name):
    path_for_ts_files = setup_directory(channel_name)
    m3u8_for_1080p = get_stream_m3u8_for_channel(channel_name)
    stream_video(m3u8_for_1080p, path_for_ts_files)

def download_and_save_vod(vod_id, channel_name):
    # technically can just get channel name from api but fuck it
    path_for_ts_files = setup_directory(channel_name)
    m3u8_url = get_stream_m3u8_url_for_vod(vod_id)
    stream_m3u8_req = requests.get(m3u8_url, headers=CHROME)
    stream_m3u8_req.raise_for_status()

    ts_file_links = get_ts_files_from_vod_m3u8_text(m3u8_url, stream_m3u8_req.text)
    download_all_ts_files(path_for_ts_files, ts_file_links)
    print(ts_file_links)


if __name__ == "__main__":
    #download_and_save_stream("gosu")
    #r = get_stream_m3u8_url_for_vod("418217865")
    t0 = time.time()
    download_and_save_vod("468178925", "shroud")
    #download_and_save_stream("overwatchleague")
    print("Total time", (time.time() - t0)/60, "minutes")


