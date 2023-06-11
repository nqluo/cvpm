import cv2
import os
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt
from scipy.io import savemat

import dlib
import glob
from skimage import io

## config the parameters
# can ONLY save one video, otherwise will decrease FPS
SAVE_VIDEO_w_label = False
SAVE_VIDEO_wo_label = False
# save the data as .mat file
SAVE_DATA = False
# show the ROI and face landmarks
SHOW_ROI = False
SHOW_FACE_LANDMARKS = True

save_rppg = True
if save_rppg:
    SAVE_VIDEO_w_label = True
    SAVE_DATA = True

## TEXT font
# font 
font = cv2.FONT_HERSHEY_SIMPLEX   
# fontScale 
fontScale = 0.5
# Green color in BGR 
color = (0, 255, 0)  
# Line thickness of 2 px 
thickness = 1


## Face detector
face_detector_path = './cascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_detector_path)
face_dlib = dlib.get_frontal_face_detector() 
## Face 81 point landmarks
face_land_mark_path = './face_landmarks/shape_predictor_81_face_landmarks.dat'
predictor = dlib.shape_predictor(face_land_mark_path)


## Camera and HR config
frame_fps = 15
time_duration_sec = 10 # FFT_RESOLUTION = 1/time_duration_sec 
time_update_sec = 1
frame_len = int(frame_fps * time_duration_sec) # FFT_RESOLUTION(Hz) = FPS / TOTAL_FRAME = 1 / TIME_ACCUMULATE_CAL_HR
hr_update_cnt = int(frame_fps * time_update_sec)


## Initia the pixel value and array, roi region
array_r, array_g, array_b, array_rPPG, array_time = [], [], [], [], []
val_r, val_g, val_b, val_rPPG = None, None, None, None
x, y, w, h = None, None, None, None
roi_x, roi_y, roi_w, roi_h = None, None, None, None
hr_cal = None
hr_cnt = 0
hr_cnt_average = 5 # ten hr average to display
array_hr = np.array([ None for _ in range(hr_cnt_average)], dtype = np.float32)
array_hr[0] = 60 # initial value
hr_avg = np.nan # display the hr_avg

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, frame_fps)
frame_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

timeString = time.strftime("%Y-%m-%d-%H-%M")

if not os.path.exists('./rppg'):
    os.mkdir('./rppg')

video_path_w_label = f'./rppg/{timeString}_w_label.mp4'
video_path_wo_label = f'./rppg/{timeString}_wo_label.mp4'
rppg_data_path = f'./rppg/{timeString}.mat'


if SAVE_VIDEO_w_label or SAVE_VIDEO_wo_label:
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    if SAVE_VIDEO_w_label:
        video_out_w_label = cv2.VideoWriter(video_path_w_label, fourcc, frame_fps, frame_size)
    if SAVE_VIDEO_wo_label:
        video_out_wo_label = cv2.VideoWriter(video_path_wo_label, fourcc, frame_fps, frame_size)


frame_cnt = 0
roi_miss_cnt = 0

startTime = time.time()
prevTime = time.time() 

while (cv2.waitKey(1) != 27):
    success, frame = camera.read()
    currTime = time.time()
    if not success:
        continue

    frame_cnt += 1
    frameTime = currTime - startTime

    # check the stability of fps
    sec = currTime - prevTime
    fps = 1./sec
    prevTime = currTime

    frame = cv2.flip(frame, 1)

    # label each frame with camera config
    cv2.putText(frame, "Time: {0:.1f} sec".format(frameTime), (50, 60), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, "FPS: {0:.0f} (set: {1:.0f}) ".format(fps, frame_fps), (50, 80), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, 'Size W*H: {}'.format(frame_size), (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)

    # Save video without label
    if SAVE_VIDEO_wo_label:
        video_out_wo_label.write(frame)

    # ## face_dlib, more stable, but much slower
    # dets = face_dlib(frame, 0) ## ***This face detector decrease the FPS from 30fps to 8fps**
    # for k, d in enumerate(dets):
    #     shape = predictor(frame, d) # 81 point landmarks
    #     cv2.rectangle(frame, (d.left(), d.top()),(d.right(),d.bottom()), (255,0,0), 1) # draw face rectangle
    #     landmarks = [[p.x, p.y] for p in shape.parts()] # 81 point landmarks   
    #     for [center_x, center_y] in landmarks:
    #         cv2.circle(frame, (center_x, center_y), 1, (0,255,0), -1) # draw 81 points landmarks
    #     ## Define the ROI
    #     [roi_x, roi_y], [roi_w, roi_h] = landmarks[69], np.array(landmarks[24]) - np.array(landmarks[69])
    #     cv2.rectangle(frame, (roi_x, roi_y),(roi_x+roi_w, roi_y+roi_h), (0,0,255), 1) # draw ROI: forehead 


    # ## face_cascade, unstable, but fast
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120)) # use the cascade face dector instead
    # for (x, y, w, h) in faces:
    #     d = dlib.rectangle(x,y,x+w,y+h)
    #     shape = predictor(frame, d) # 81 point landmarks
    #     # cv2.rectangle(frame, (d.left(), d.top()),(d.right(),d.bottom()), (255,0,0), 1) # draw face rectangle  
    #     landmarks = [[p.x, p.y] for p in shape.parts()] # 81 point landmarks   
    #     for [center_x, center_y] in landmarks:
    #         cv2.circle(frame, (center_x, center_y), 1, (0,255,0), -1) # draw 81 points landmarks
    #     ## Define the ROI
    #     [roi_x, roi_y], [roi_w, roi_h] = landmarks[69], np.array(landmarks[24]) - np.array(landmarks[69])
    #     cv2.rectangle(frame, (roi_x, roi_y),(roi_x+roi_w, roi_y+roi_h), (0,0,255), 1) # draw ROI: forehead 


    ## Select the first subject to process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120, 120)) # use the cascade face dector instead
    
    # update ROI value if applicable
    if len(faces) >= 1:
        # if x == None:
        x, y, w, h = faces[0]
        d = dlib.rectangle(x,y,x+w,y+h)
        shape = predictor(frame, d) # 81 point landmarks
        landmarks = [[p.x, p.y] for p in shape.parts()] # 81 point landmarks  
        # Define the ROI
        [roi_x, roi_y], [roi_w, roi_h] = landmarks[69], np.array(landmarks[24]) - np.array(landmarks[69])
        roi_x, roi_y, roi_w, roi_h = roi_shrink(roi_x, roi_y, roi_w, roi_h, keep = 0.8)
        # Extract the ROI
        roi_rgb = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] # update forehead
        val_r = np.average(roi_rgb[:,:,2])
        val_g = np.average(roi_rgb[:,:,1])
        val_b = np.average(roi_rgb[:,:,0]) 


        if SHOW_ROI:
            cv2.imshow('ROI', roi_rgb)
        # Draw the landmarks
        if SHOW_FACE_LANDMARKS:
            # cv2.rectangle(frame, (d.left(), d.top()),(d.right(),d.bottom()), (255,0,0), 1) # draw face rectangle       
            for [center_x, center_y] in landmarks:
                cv2.circle(frame, (center_x, center_y), 1, (0,255,0), -1) # draw 81 points landmarks
            cv2.rectangle(frame, (roi_x, roi_y),(roi_x+roi_w, roi_y+roi_h), (0,0,255), 1) # draw ROI: forehead 

        # label RGB avg
        cv2.putText(frame, "R_avg: {:.2f}".format(val_r), (50, 140), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "G_avg: {:.2f}".format(val_g), (50, 160), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, "B_avg: {:.2f}".format(val_b), (50, 180), font, fontScale, color, thickness, cv2.LINE_AA)

    else:
        if array_g:
            val_r = np.average(array_r[-5:])
            val_g = np.average(array_g[-5:])
            val_b = np.average(array_b[-5:])
            roi_miss_cnt += 1


    ## Accumulate the array_r/g/b/rPPG
    if val_g:
        array_r.append(val_r)
        array_g.append(val_g)
        array_b.append(val_b)
        array_time.append(frameTime)

    # update the hr_cal is applicable
    if frame_cnt % hr_update_cnt == 0:
        cal_r = array_r[-frame_len:]
        cal_g = array_g[-frame_len:]
        cal_b = array_b[-frame_len:]
        cal_t = array_time[-frame_len:]
        rPPG = pos_rppg(cal_r, cal_g, cal_b, frame_fps)
        hr_cal = cal_hr(rPPG, cal_t)
        hr_cnt += 1
        array_hr[int(hr_cnt%hr_cnt_average)] = hr_cal
        hr_avg = np.nanmean(array_hr)

            # hr_cal = None
        # elif roi_miss_cnt >= hr_update_cnt:
        #     hr_cal = None
        roi_miss_cnt = 0 # restart the roi_miss_cnt


    # cv2.putText(frame, "HR:{:.0f}BPM".format(hr_avg), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, "HR:{:.0f}BPM".format(hr_avg), (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    # cv2.putText(frame, "HR: {}".format(array_hr), (50, 430), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.imshow('Face Detection', frame)

    # Save video with label
    if SAVE_VIDEO_w_label:
        video_out_w_label.write(frame)

cv2.destroyAllWindows()
if SAVE_VIDEO_w_label:
    video_out_w_label.release()

if SAVE_VIDEO_wo_label:
    video_out_wo_label.release()

camera.release()

# save the rPPG data
if SAVE_DATA:
    data = {}
    data['r'] = np.array(array_r)
    data['g'] = np.array(array_g)
    data['b'] = np.array(array_b)
    data['time'] = np.array(array_time)
    data['FPS'] = frame_fps
    rPPG_POS_all = pos_rppg(array_r, array_g, array_b, frame_fps)
    # rPPG_POS_all = ppg_filter(rPPG_POS_all, frame_fps)
    data['rPPG'] = rPPG_POS_all
    savemat(rppg_data_path, data)
    print('rPPG saved to {}'.format(rppg_data_path))