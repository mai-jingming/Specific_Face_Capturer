# Author: https://www.dfldata.xyz/forum.php Gottvonkarlberg
# github: https://github.com/mai-jingming/Specific_Face_Capturer

import mediapipe as mp
import numpy as np
import os
import face_recognition
import time
from decord import VideoReader
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar


class SFC:
    def __init__(self, vid_path, target_dir, save_dir, interval=1, tolerance=0.6):
        self.vid_path = vid_path
        self.target_dir = target_dir
        self.save_dir = save_dir
        self.interval = interval
        self.tolerance = tolerance
        self.tg_encodings = []
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                                        min_detection_confidence=0.5)

    def tg_encoder(self):
        assert len(os.listdir(self.target_dir)) != 0, "需要在 target_face 文件夹里放上要找的人的图片"

        for file in os.listdir(self.target_dir):
            file_path = os.path.join(self.target_dir, file)
            img = face_recognition.load_image_file(file_path)
            tg_face_encoding = face_recognition.face_encodings(img, model='large')[0]
            self.tg_encodings.append(tg_face_encoding)

    def postprocessing(self, y_min, x_max, y_max, x_min, W, H):
        x_min = x_min - 100 * W / 1920
        y_min = y_min - 200 * H / 1080
        x_max = x_max + 100 * W / 1920
        y_max = y_max + 150 * H / 1080
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max >= W:
            x_max = W - 1
        if y_max >= H:
            y_max = H - 1
        return int(y_min), int(x_max), int(y_max), int(x_min)

    def face_locating(self, frame, W, H):
        face_locations = []
        results = self.face_detection.process(frame)
        if results.detections:
            for det in results.detections:
                rbb = det.location_data.relative_bounding_box
                if rbb.width * W < 256 or rbb.height * H < 256:
                    continue
                x_min = rbb.xmin * W
                y_min = rbb.ymin * H
                x_max = (rbb.width + rbb.xmin) * W
                y_max = (rbb.height + rbb.ymin) * H
                face_locations.append((int(y_min), int(x_max), int(y_max), int(x_min)))
        return face_locations

    def process(self):
        # 在连续读取视频帧上，decord速度是opencv的2.28倍
        # 在存储图片到硬盘上，matplotlib速度是opencv的1.146倍
        self.tg_encoder()
        vr = VideoReader(self.vid_path)
        frame_num = len(vr)
        print("总帧数为：", frame_num)
        height, width = vr[0].shape[0], vr[0].shape[1]
        print(f'视频分辨率为：{width}×{height}')
        bar = IncrementalBar('Processing', max=frame_num // interval)
        for frame_id in range(0, frame_num, self.interval):
            frame = vr[frame_id]
            if not frame:
                continue
            frame = frame.asnumpy()
            face_locations = self.face_locating(frame, width, height)

            if face_locations:
                face_encodings = face_recognition.face_encodings(frame, face_locations, model='large')
                for face_encoding in self.tg_encodings:
                    matches = face_recognition.compare_faces(face_encoding, face_encodings, tolerance=self.tolerance)
                    face_distances = face_recognition.face_distance(face_encoding, face_encodings)
                    if len(face_distances) == 0:
                        continue
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        y_min, x_max, y_max, x_min = face_locations[best_match_index]
                        y_min, x_max, y_max, x_min = self.postprocessing(y_min, x_max, y_max, x_min, width, height)
                        face = frame[y_min:y_max, x_min:x_max]
                        pth = os.path.join(self.save_dir, str(frame_id) + '.jpg')
                        plt.imsave(pth, face)
            bar.next()
        bar.finish()


if __name__ == '__main__':
    print("作者：Gottvonkarlberg\n\n")
    print("请先在任意位置创建一个名叫 target_face 的文件夹，然后在里面放入目标视频中截取的目标人物的脸至少一张\n")
    print("请在 target_face 文件夹所在的目录也创建一个叫 saved_face 的文件夹以便在同一个目录里查看结果\n")
    target_path = input("请将刚刚创建的 target_face 文件夹拖入命令框，出现文件夹路径后按 Enter 键继续\n")
    print('')
    save_dir = input("请将刚刚创建的 saved_face 文件夹拖入命令框，出现文件夹路径后按 Enter 键继续\n")
    print('')
    vid_path = input("请将目标视频拖入命令框，出现文件路径后按 Enter 键继续\n")
    print('')
    tolerance = input("请输入筛选阈值，取值范围为[0.0-1.0]，数值越小筛选越严格，越容易漏选；数值越大筛选越宽松，越容易选到其他人，推荐0.6\n")
    print('')
    tolerance = float(tolerance)
    interval = input('请输入采样间隔，即每多少帧采样一次，用于跳过连续重复的帧。不能超过总帧数, 不能小于1\n')
    print('')
    interval = int(interval)
    sfc = SFC(vid_path, target_dir=target_path, save_dir=save_dir, interval=interval, tolerance=tolerance)
    start = time.time()
    sfc.process()
    end = time.time()
    print(f'总耗时为：{end-start} 秒\n')
    input("按 Enter 键退出")
