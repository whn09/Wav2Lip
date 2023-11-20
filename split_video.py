import subprocess
import math
import os

def split_video(video_file, output, duration):
    # 获取视频的总长度（以秒为单位）
    cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}".format(video_file)
    total_time = float(subprocess.check_output(cmd, shell=True))

    # 计算需要分割的段数
    num_segments = math.ceil(total_time / duration)

    os.makedirs(output, exist_ok=True)

    # 使用ffmpeg将视频分割成多段
    for i in range(num_segments):
        start = i * duration
        cmd = "ffmpeg -i {} -ss {} -t {} {}/{}.mp4".format(video_file, start, duration, output, i)  # -c copy 
        subprocess.call(cmd, shell=True)

# 用法示例
split_video("../inputs/train.mp4", "../inputs/train", 1)
