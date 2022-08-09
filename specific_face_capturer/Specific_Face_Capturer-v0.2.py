# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 05:49:15 2022

# Author: https://www.dfldata.xyz/forum.php Gottvonkarlberg
# github: https://github.com/mai-jingming/Specific_Face_Capturer
"""
import face_recognition
import time
import cv2
import os
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def preprocessing(frame):
    # 对原图做缩小处理，提高推演效率
    small_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    rgb_small_frame = small_frame[:,:,::-1]
    return rgb_small_frame

def postprocessing(rgb_small_frame):
    # 将缩小的图变换为原图大小
    small_frame = rgb_small_frame[:,:,::-1]
    frame = cv2.resize(small_frame,(0,0),fx=2,fy=2)
    return frame

def known_face_encoding(known_face_path):
    # 提取要比对的脸的特征
    known_face_encoding_list = []
    assert len(os.listdir(known_face_path)) != 0, "需要在 target_face 文件夹里放上要找的人的图片"
    
    for file in os.listdir(known_face_path):
        file_path = os.path.join(known_face_path,file)
        img = face_recognition.load_image_file(file_path)
        small_img = cv2.resize(img,(0,0),fx=1,fy=1)
        known_face_encoding = face_recognition.face_encodings(small_img)[0]
        known_face_encoding_list.append(known_face_encoding)
    return known_face_encoding_list

def face_matching(rgb_small_frame, known_face_encoding_list, frame_id, tolerance):
    # 找一个帧里的所有脸的位置并提取其特征编码
    face_locations = face_recognition.face_locations(rgb_small_frame, 0, model='cnn') # (top, right, bottom, left) order
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model='large')
    
    # 对上述找到的脸做一个匹配
    for face_encoding in known_face_encoding_list:
        matches = face_recognition.compare_faces(face_encoding, face_encodings,tolerance=tolerance)
        face_distances = face_recognition.face_distance(face_encoding, face_encodings)
        if len(face_distances) == 0:
            continue
        best_match_index = np.argmin(face_distances)
    
        if matches[best_match_index]:
            (top, right, bottom, left) = face_locations[best_match_index]
            if (bottom-top)>=128 and (right-left)>=128:
                top = top * 2 - 100
                right = right * 2 + 100
                bottom = bottom * 2 + 100
                left = left * 2 - 100
                frame = postprocessing(rgb_small_frame)
                if top < 0:
                    top = 0
                if right >= frame.shape[1]:
                    right = frame.shape[1] - 1
                if bottom >= frame.shape[0]:
                    bottom = frame.shape[0] - 1
                if left < 0:
                    left = 0
                face = frame[top:bottom+1,left:right+1]
                print("第 %d 帧找到了目标脸" %frame_id)
                return face
    
    print("第 %d 帧没有找到目标脸" %frame_id)
    return np.empty(0)

def save_faces(face, save_path, frame_id):
    if face.size != 0:
        face_path = os.path.join(save_path, str(frame_id) + ".jpg")
        cv2.imwrite(face_path, face)

class Process:
    def __init__(self, video_path, tolerance, batch_size=2):
        self.video_path = video_path
        self.step = 0 # 每一组里要处理的帧数
        self.num_processors = mp.cpu_count() if mp.cpu_count() <= batch_size else batch_size
        self.known_face_path = "target_face"
        self.save_path = "saved_face"
        self.tolerance = tolerance
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def single_process(self, frame_group_index):
        known_face_encoding_list = known_face_encoding(self.known_face_path)
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = self.step * frame_group_index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # 设置要获取的帧组的首索引
        num_processed_frame = 0
    
        if frame_group_index != self.num_processors-1:
            not_last_group = True
        else:
            not_last_group = False
        
        while cap.isOpened():
            if not_last_group and num_processed_frame == self.step:
                break
            ret, frame = cap.read()
            if not ret:
                break
            rgb_small_frame = preprocessing(frame)
            frame_id = frame_idx + num_processed_frame
            face = face_matching(rgb_small_frame, known_face_encoding_list, frame_id, tolerance=self.tolerance)
            save_faces(face, self.save_path, frame_id)
            num_processed_frame += 1
            
        cap.release()

    def multi_process(self):
        # 将视频的帧切分成最大进程数个视频帧组，然后每一个进程处理一个视频帧组
        # 对 1080p 24FPS 1min时长的视频进行测试：
        # 单进程：57秒
        # 双进程：39秒
        # 三进程：38秒
        # 四进程：83秒
        cap = cv2.VideoCapture(self.video_path)
        num_frames = int(cap.get(7)) # 返回视频总帧数，返回值为float类型
        self.step = num_frames // self.num_processors
        cap.release()
        
        with ProcessPoolExecutor(max_workers=self.num_processors) as ppool:
            ppool.map(self.single_process, range(self.num_processors))
    
if __name__ == "__main__":
    print("请先在此程序所在文件夹创建一个名叫 target_face 的文件夹，然后在里面放入目标视频中截取的目标人物的脸至少一张，角度越多越不容易漏选")
    dir_exist = input("请问是否已创建 target_face 文件夹？ [y/n]\n")
    while not dir_exist == 'y' and  not dir_exist == 'Y':
        print("请先创建 target_face 文件夹")
        dir_exist = input("请问是否已创建 target_face 文件夹？ [y/n]\n")
    print("请先在此程序所在文件夹放入你的目标视频")
    video_path = input("请输入目标视频文件名称，如：“vid01.mp4”，不要用中文！\n")
    while not os.path.exists(video_path):
        print("目标视频不存在或输入的视频名字有误")
        video_path = input("请输入目标视频文件名称，如：“vid01.mp4”，不要用中文！\n")
    tolerance = input("请输入筛选阈值，取值范围为[0.0-1.0]，数值越小筛选越严格，越容易漏选；数值越大筛选越宽松，越容易选到其他人，推荐0.5\n")
    tolerance = float(tolerance)
    start = time.time()
    p = Process(video_path, tolerance)
    p.multi_process()
    end = time.time()
    print("总耗时为: %f s" % (end-start))
