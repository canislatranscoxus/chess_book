import numpy as  np
import cv2

video_path = '/home/art/Pictures/yolo/video_sample2.mp4'
file_video_stream = cv2.VideoCapture( video_path )

#create a while loop
while (file_video_stream.isOpened):
    #get the current frame from video stream
    ret,current_frame = file_video_stream.read()
    #use the video current frame instead of image
    #img_to_detect = current_frame

    cv2.imshow("p1", current_frame)
    print( '...' )