import os
import json
import numpy as np
import cv2
import torch
import boto3
import tempfile
import time
from tqdm import tqdm

import sys
sys.path.append('..')
import audio
import face_detection
from models import Wav2Lip


s3 = boto3.client('s3')

bucket_name = 'sagemaker-us-east-1-579019700964'
prefix = 'sagemaker/wav2lip'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

static = False
pads = [0, 10, 0, 0]
face_det_batch_size = 8
wav2lip_batch_size = 8
resize_factor = 1
crop = [0, -1, 0, -1]
box = [-1, -1, -1, -1]
rotate =False
nosmooth = False

mel_step_size = 16
img_size = 96


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()


def model_fn(model_dir):
    model = load_model(os.path.join(model_dir, 'wav2lip_gan.pth'))
    print("Model loaded")
    return model


def parse_s3_url(url):
    parts = url.replace("s3://", "").split("/")
    bucket = parts.pop(0)
    key = "/".join(parts)
    return bucket, key


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

    batch_size = face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
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
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def input_fn(request_body, request_content_type):
    start_time = time.time()
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

    video_path = input_data['video_path']
    audio_path = input_data['audio_path']

    # 从S3下载视频文件
    video_bucket, video_key = parse_s3_url(video_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        s3.download_fileobj(video_bucket, video_key, f)
        video_file = f.name

    # 读取视频文件并提取视频帧
    video_stream = cv2.VideoCapture(video_file)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

        if rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)

    print("Number of frames available for inference: "+str(len(full_frames)))

    # 从S3下载音频文件
    bucket, key = parse_s3_url(audio_path)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        s3.download_fileobj(bucket, key, f)
        audio_file = f.name
    
    # 对视频帧和音频数据进行预处理,提取特征等
    wav = audio.load_wav(audio_file, 16000)
    mel = audio.melspectrogram(wav)

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

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    if box[0] == -1:
        if not static:
            face_det_results = face_detect(full_frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([full_frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]
    for i in range(len(face_det_results)):
        face_det_results[i][0] = cv2.resize(face_det_results[i][0], (img_size, img_size))

    # 删除临时视频文件和音频文件
    os.unlink(video_file)
    # os.unlink(audio_file)
    
    end_time = time.time()
    print('input_fn time:', end_time-start_time)

    return audio_file, fps, full_frames, mel_chunks, face_det_results


def datagen(frames, mels, face_det_results):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = 0 if static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def upload_to_s3(filename):
    # 指定上传到S3的对象键
    object_key = prefix+'/output/'+filename.split('/')[-1]
    
    # 使用临时文件作为上传的缓冲区
    output_url = ''
    try:
        # 直接将视频文件上传到S3
        s3.upload_file(filename, bucket_name, object_key)

        print(f"File uploaded to S3: s3://{bucket_name}/{object_key}")

        # 返回S3对象的URL或其他所需的信息
        output_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
    except Exception as e:
        print(f"Error uploading AVI file to S3: {str(e)}")

    return output_url

    
def predict_fn(input_data, model):
    audio_file, fps, full_frames, mel_chunks, face_det_results = input_data
    output_file = audio_file+'.avi'

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames, mel_chunks, face_det_results)

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    start_time = time.time()
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    end_time = time.time()
    print('Wav2Lip inference time:', end_time-start_time)

    # command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file, output_file, audio_file+'.mp4')
    # subprocess.call(command, shell=platform.system() != 'Windows')
    
    output_url = upload_to_s3(output_file)

    # 删除临时视频文件和音频文件
    os.unlink(audio_file)
    os.unlink(output_file)

    return output_url


def output_fn(prediction, content_type):
    result = {'output_path': prediction}
    return result


# if __name__ == '__main__':

#     model = model_fn('../')

#     request_body = json.dumps({'video_path': 's3://sagemaker-us-east-1-579019700964/sagemaker/wav2lip/train_close_mouth_480p.mp4', 'audio_path': 's3://sagemaker-us-east-1-579019700964/sagemaker/wav2lip/test.wav'})
#     request_content_type = 'application/json'
#     input_data = input_fn(request_body, request_content_type)

#     prediction = predict_fn(input_data, model)

#     response_content_type = 'application/json'
#     result = output_fn(prediction, response_content_type)
#     print('result:', result)