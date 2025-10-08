import sys
from collections import namedtuple
import cv2
import numpy as np

#Instalar: pip install opencv-python opencv-contrib-python numpy
# Importar las clases de los otros archivos
from ejer10_1 import ROISelector
from ejer10_2 import PoseEstimator


class VideoHandler(object):
    def __init__(self, capId, scaling_factor, win_name):
        self.cap = cv2.VideoCapture(capId)
        self.pose_tracker = PoseEstimator()
        self.win_name = win_name
        self.scaling_factor = scaling_factor
        ret, frame = self.cap.read()
        self.rect = None
        self.frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        self.roi_selector = ROISelector(win_name, self.frame, self.set_rect)

    def set_rect(self, rect):
        self.rect = rect
        self.pose_tracker.add_target(self.frame, rect)
    
    def start(self):
        paused = False
        while True:
            if not paused or self.frame is None:
                ret, frame = self.cap.read()
                scaling_factor = self.scaling_factor
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                if not ret:
                    break
                self.frame = frame.copy()
            
            img = self.frame.copy()
            
            if not paused and self.rect is not None:
                tracked = self.pose_tracker.track_target(self.frame)
                for item in tracked:
                    cv2.polylines(img, [np.int32(item.quad)], True, (255, 255, 255), 2)
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(img, (x, y), 2, (255, 255, 255))
            
            self.roi_selector.draw_rect(img, self.rect)
            cv2.imshow(self.win_name, img)
            ch = cv2.waitKey(1)
            
            if ch == ord(' '):
                paused = not paused
            if ch == ord('c'):
                self.pose_tracker.clear_targets()
            if ch == 27:
                break


if __name__ == '__main__':
    VideoHandler(0, 0.8, 'Tracker').start()