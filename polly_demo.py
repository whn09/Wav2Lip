import boto3
from io import BytesIO
import numpy as np
import librosa
import wave

polly_client = boto3.Session().client('polly')

# response = polly_client.synthesize_speech(VoiceId='Joanna',
#                 OutputFormat='mp3', 
#                 Text = 'This is a sample text to be synthesized.',
#                 Engine = 'neural')

# file = open('speech.mp3', 'wb')
# file.write(response['AudioStream'].read())
# file.close()

response = polly_client.synthesize_speech(VoiceId='Joanna',
                OutputFormat='pcm', 
                Text = 'This is a sample text to be synthesized.')


# 从响应中获取 PCM 数据流
pcm_stream = response['AudioStream']

# 用于存储所有的 PCM 数据
pcm_data = b''

# 读取 PCM 数据
while True:
    chunk = pcm_stream.read(1024)
    if not chunk:
        break
    pcm_data += chunk

    # 将 PCM 数据转换为 numpy 数组
    chunk_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    # 将 PCM 数据范围从 [-32768, 32767] 转换为 [-1.0, 1.0]
    # chunk_array /= 32768.0
    # 使用 librosa 生成 Mel 频谱
    mel_spectrogram = librosa.feature.melspectrogram(y=chunk_array, sr=16000, n_fft=400, hop_length=160, n_mels=80, fmin=50, fmax=7600)
    # mel spectrogram 可能在 dB scale, 你可能需要转换它
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # 这样你就得到了 mel 频谱
    print(log_mel_spectrogram)

# 将 PCM 数据转换为 numpy 数组
pcm_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
# 将 PCM 数据范围从 [-32768, 32767] 转换为 [-1.0, 1.0]
# pcm_array /= 32768.0
# 使用 librosa 生成 Mel 频谱
mel_spectrogram = librosa.feature.melspectrogram(y=pcm_array, sr=16000, n_fft=400, hop_length=160, n_mels=80, fmin=50, fmax=7600)
# mel spectrogram 可能在 dB scale, 你可能需要转换它
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
# 这样你就得到了 mel 频谱
print(log_mel_spectrogram)

# 创建一个新的 WAV 文件
with wave.open('output.wav', 'w') as wav_file:
    # 设置音频参数
    # nchannels：声道数，这里设为1，表示单声道
    # sampwidth：每个样本的字节数，这里设为2，表示16位PCM
    # framerate：采样率，这里设为16000，表示16kHz
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)

    # 写入 PCM 数据
    wav_file.writeframes(pcm_data)