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

# text = '''
# It has been said that everyone lives in two worlds -- the world outside and the world inside. The world outside is the world of events, people, and things. It is the world we share with others, the world we can see, hear, touch, and taste. The world inside is the world of our own private thoughts, feelings, and dreams. It is the world we know best but find hard to explain or share with others.
# These two worlds are closely connected. What happens in one world influences what happens in the other world. If we are unhappy in the world outside, our private world also tends to be unhappy. If we are contented in our private world, then we tend to be more tolerant and understanding in our contacts with the world outside. If a man loses his job and his income, it not only means that he has less money to spend, but it may also mean that he loses confidence in himself. The world outside has invaded the world inside. Almost everything that happens to us in the world outside has some effect on our private world.
# '''

text = '''
有人说，每个人都生活在两个世界中——外部世界和内部世界。 外面的世界是由事件、人和事组成的世界。 这是我们与他人共享的世界，是我们能看到、听到、触摸到、尝到的世界。 内在的世界是我们自己私人的思想、感情和梦想的世界。 这是我们最了解但很难解释或与他人分享的世界。
这两个世界紧密相连。 一个世界发生的事情会影响另一个世界发生的事情。 如果我们在外面的世界不快乐，我们的私人世界也往往会不快乐。 如果我们对自己的私人世界感到满足，那么我们在与外界的接触中往往会更加宽容和理解。 如果一个人失去了工作和收入，这不仅意味着他可以花的钱减少了，还可能意味着他对自己失去了信心。 外面的世界已经入侵了里面的世界。 几乎我们在外面世界发生的所有事情都会对我们的私人世界产生一些影响。
'''

response = polly_client.synthesize_speech(VoiceId='Zhiyu',  # 'Joanna'
                OutputFormat='pcm', 
                Text = text)  # 'This is a sample text to be synthesized.'


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
    # print(log_mel_spectrogram)

# 将 PCM 数据转换为 numpy 数组
pcm_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
# 将 PCM 数据范围从 [-32768, 32767] 转换为 [-1.0, 1.0]
# pcm_array /= 32768.0
# 使用 librosa 生成 Mel 频谱
mel_spectrogram = librosa.feature.melspectrogram(y=pcm_array, sr=16000, n_fft=400, hop_length=160, n_mels=80, fmin=50, fmax=7600)
# mel spectrogram 可能在 dB scale, 你可能需要转换它
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
# 这样你就得到了 mel 频谱
# print(log_mel_spectrogram)

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