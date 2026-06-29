 
 
import os
import argparse
from moviepy import *

def extract_audio_from_mp4(mp4_file, wav_file):
    # 打开MP4文件
    video = VideoFileClip(mp4_file)
    
    try:
        # 读取音频数据
        audio = video.audio
        
        # 将音频数据保存为WAV文件
        audio.write_audiofile(wav_file)
    finally:
        # 关闭MP4文件
        video.close()

 
 

if __name__ == "__main__":
    mp4_path="F:\\video\\31d3ffc03ccc7ba5fb16f40a600b55c1.mp4"
    wav_path="tmp.wav"
    extract_audio_from_mp4(mp4_path, wav_path)
 