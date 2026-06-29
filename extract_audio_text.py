from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from convert_mp42wav import *
from pydub import AudioSegment
 
wav_path="tmp.wav"

inference_pipeline = pipeline(task=Tasks.auto_speech_recognition,model='F:\\Whisper'  )


def split_wav(input_file: str,  segment_length: int = 15, overlap: int = 0):
 
    audio = AudioSegment.from_wav(input_file)
 
    
    # 获取基本文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 计算每个片段的毫秒数
    segment_length_ms = segment_length * 1000
    overlap_ms = overlap * 1000
    step_ms = segment_length_ms - overlap_ms
    
    # 获取音频总时长
    duration_ms = len(audio)
    result=""
    # 切割音频
    segment_count = 0
    for start_ms in range(0, duration_ms, step_ms):
        end_ms = start_ms + segment_length_ms   
        # 确保不超出音频边界
        if end_ms > duration_ms:
            end_ms = duration_ms  
        # 提取片段
        segment = audio[start_ms:end_ms]
        # 保存片段
        segment.export("tmp2.wav", format="wav")
        #提取语音转成的文字
        rec_result = inference_pipeline(input="tmp2.wav", language=None )
        result+=rec_result[0]["text"]
        segment_count += 1
    return result
def extract_audio_text(mp4_path):
    extract_audio_from_mp4(mp4_path, wav_path)
    result=split_wav(wav_path,  segment_length= 15, overlap = 0)
    return result


#rec_result = inference_pipeline(input=wav_path, language=None )
#print(rec_result)