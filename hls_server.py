# pip install ffmpeg-python flask flask-cors
# python hls_server.py

import os
import time
import ffmpeg
from flask import Flask, request, send_from_directory, make_response
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)  # 这将为所有路由启用 CORS

VIDEO_DIR = "received_videos"
HLS_DIR = "www/hls"
SEGMENT_TIME = 4  # 1秒一段
MAX_SEGMENT_TIME = 4

total_duration = 0  # 添加一个全局变量来跟踪总时长

def get_video_duration(file_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def create_hls_segment(video_file):
    global total_duration
    
    duration = get_video_duration(os.path.join(VIDEO_DIR, video_file))
    print('duration:', duration)
    
    output = os.path.join(HLS_DIR, f'segment{int(time.time()*1000)}.ts')
    
    stream = ffmpeg.input(os.path.join(VIDEO_DIR, video_file))
    stream = ffmpeg.output(stream, output, vcodec='h264', acodec='aac', output_ts_offset=total_duration)  # , hls_time=SEGMENT_TIME, hls_list_size=0
    ffmpeg.run(stream, overwrite_output=True)
    
    total_duration += duration
    
    return os.path.basename(output), duration

def update_playlist(segment, duration):
    playlist_path = os.path.join(HLS_DIR, 'playlist.m3u8')
    
    with open(playlist_path, 'a') as f:
        # f.write(f'#EXTINF:{SEGMENT_TIME},\n')
        f.write(f'#EXTINF:{duration},\n')
        f.write(f'{segment}\n')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = os.path.join(VIDEO_DIR, file.filename)
        file.save(filename)
        segment, duration = create_hls_segment(file.filename)
        update_playlist(segment, duration)
        return 'File uploaded successfully', 200

@app.route('/hls/<path:filename>')
def serve_hls(filename):
    return send_from_directory(HLS_DIR, filename)
    # response = make_response(send_from_directory(HLS_DIR, filename))
    # response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    # response.headers['Pragma'] = 'no-cache'
    # response.headers['Expires'] = '0'
    # return response

if __name__ == '__main__':
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    if not os.path.exists(HLS_DIR):
        os.makedirs(HLS_DIR)

    # 初始化播放列表
    with open(os.path.join(HLS_DIR, 'playlist.m3u8'), 'w') as f:
        f.write('#EXTM3U\n')
        f.write('#EXT-X-VERSION:3\n')
        # f.write(f'#EXT-X-TARGETDURATION:{SEGMENT_TIME}\n')
        f.write(f'#EXT-X-TARGETDURATION:{MAX_SEGMENT_TIME}\n')
        f.write('#EXT-X-MEDIA-SEQUENCE:0\n')

    context = ('/home/ubuntu/VITA/web_demo/vita_html/web/resources/cert.pem', '/home/ubuntu/VITA/web_demo/vita_html/web/resources/key.pem')
    app.run(host='0.0.0.0', port=5000, ssl_context=context)