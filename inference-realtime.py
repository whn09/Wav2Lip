import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
# from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import audio
import platform

import time
import boto3


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)
# parser.add_argument('--audio', type=str, 
#                     help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--text', type=str, 
                    help='Text used to synthesize speech', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

# 创建Amazon Polly客户端
polly_client = boto3.Session().client('polly')

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            # for i in tqdm(range(0, len(images), batch_size)):
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            # print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, face_det_results, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # if args.box[0] == -1:
    #     if not args.static:
    #         face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
    #     else:
    #         face_det_results = face_detect([frames[0]])
    # else:
    #     print('Using the specified bounding box instead of face detection...')
    #     y1, y2, x1, x2 = args.box
    #     face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        # face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    # print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        # print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    # print ("Number of frames available for inference: "+str(len(full_frames)))

    face_det_results = face_detect(full_frames)
    for i in range(len(face_det_results)):
        face_det_results[i][0] = cv2.resize(face_det_results[i][0], (args.img_size, args.img_size))

    # if not args.audio.endswith('.wav'):
    #     print('Extracting raw audio...')
    #     command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

    #     subprocess.call(command, shell=True)
    #     args.audio = 'temp/temp.wav'

    model = load_model(args.checkpoint_path)
    # print ("Model loaded")

    response = polly_client.synthesize_speech(VoiceId='Matthew',  # Joanna: Female, Matthew: Male
                    OutputFormat='pcm', 
                    Text = args.text)
                    
    pcm_stream = response['AudioStream']
    pcm_data = b''
    chunk_num = 0
    all_start_time = time.time()
    while True:
        chunk = pcm_stream.read(6096)
        if not chunk:
            break
        pcm_data += chunk

        # 将 PCM 数据转换为 numpy 数组
        chunk_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        # 将 PCM 数据范围从 [-32768, 32767] 转换为 [-1.0, 1.0]
        chunk_array /= 32768.0
        wav = chunk_array
        mel = audio.melspectrogram(wav)
        # print('mel.shape:', mel.shape)

        temp_wav_filename = 'temp/'+args.outfile.split('/')[-1][:-4]+'_chunk_{}.wav'.format(chunk_num)
        audio.save_wav(chunk_array, temp_wav_filename, 16000)
    
    # # 将 PCM 数据转换为 numpy 数组
    # pcm_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    # # 将 PCM 数据范围从 [-32768, 32767] 转换为 [-1.0, 1.0]
    # pcm_array /= 32768.0

    # audio.save_wav(chunk_array, 'temp/output.wav', 16000)

    # # wav = audio.load_wav('temp/output.wav', 16000)
    # wav = pcm_array
    # mel = audio.melspectrogram(wav)
    # print('mel.shape:', mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        # print("Length of mel chunks: {}".format(len(mel_chunks)))

        if len(mel_chunks) < 2:  # TODO pcm_stream.read(6096)的时候，len(mel_chunks)应该等于2，后面的程序运行都正常，但是最后一点语音可能不到2，导致后面的程序运行失败
            break

        sub_frames = full_frames[:len(mel_chunks)]  # TODO 暂时每个chunk只取full_frames的前一些帧，而不是顺序取
        sub_face_det_results = face_det_results[:len(mel_chunks)]

        batch_size = args.wav2lip_batch_size
        gen = datagen(sub_frames.copy(), sub_face_det_results.copy(), mel_chunks)

        frame_h, frame_w = sub_frames[0].shape[:-1]
        temp_avi_filename = 'temp/'+args.outfile.split('/')[-1][:-4]+'.avi'
        # out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        out = cv2.VideoWriter(temp_avi_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        start_time = time.time()
        avg_time = 0
        num_batches = 0
        num_frames = 0
        # for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
        #                                         total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            start = time.time()
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            # print('img_batch:', img_batch.shape, 'mel_batch:', mel_batch.shape)  # img_batch: torch.Size([1, 6, 96, 96]) mel_batch: torch.Size([1, 1, 80, 16])

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)
                num_frames += 1
            end = time.time()
            avg_time += end-start
            num_batches += 1

        out.release()
        end_time = time.time()
        # print('Wav2Lip inference time:', end_time-start_time)
        # print('Wav2Lip inference avg_time:', avg_time/num_batches)
        # print('num_batches:', num_batches)
        print('num_frames:', num_frames)

        out_mp4_filename = args.outfile[:-4]+'_chunk_'+str(chunk_num)+args.outfile[-4:]
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {} -loglevel quiet'.format(temp_wav_filename, temp_avi_filename, out_mp4_filename)
        subprocess.call(command, shell=platform.system() != 'Windows')
        rtmp_command = 'ffmpeg -re -i {} -vcodec h264 -vprofile baseline -acodec aac -ar 16000 -strict -2 -ac 1 -f flv -s 480x636 -q 10 YOUR_RTMP_URL'.format(out_mp4_filename)
        subprocess.call(rtmp_command, shell=platform.system() != 'Windows')

        chunk_num += 1
    all_end_time = time.time()
    print('chunk_num:', chunk_num)
    print('all time:', all_end_time-all_start_time)

if __name__ == '__main__':
    main()
