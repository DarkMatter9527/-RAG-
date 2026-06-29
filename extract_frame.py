# 视频信息的读取

import cv2
import numpy as np
##提取视频关键帧
def extract_keyframes(video_path, topK=5):
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    results=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #每一帧的灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算帧间差异（直方图或像素差值）
        diff = cv2.absdiff(prev_gray, gray)
        # 镜头切换、大幅动作时 diff_mean 会显著升高,定义为关键正
        diff_mean = np.mean(diff)
        results.append([frame,diff_mean])
        prev_gray = gray
    cap.release()
    results=sorted(results,key=lambda s:s[1],reverse=True)[0:topK]
    return [s[0] for s in results]
if __name__ == "__main__":
    path="F:\\video\\20f7233defd962f4c5ab25ee2d1074cc.mp4"
    frames=extract_keyframes(path)
    print (frames)