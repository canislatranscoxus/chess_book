# -*- coding: utf-8 -*-
"""

@author: Canis
"""

import numpy as  np
import cv2

class Yolo_video:

    #get the webcam video stream
    video_path          = '/home/art/Pictures/yolo/video_sample2.mp4'

    class_labels        = None
    class_colors        = None

    yolo_model          = None
    yolo_output_layer   = None
    cfg_path            = '/home/art/git/chess_book/src/yolo_01/model/yolov3.cfg'
    weights_path        = '/home/art/git/chess_book/src/yolo_01/model/yolov3.weights'

    class_ids_list      = []
    boxes_list          = []
    confidences_list    = []

    def create_video_stream( self ):
        try:
            self.file_video_stream = cv2.VideoCapture( self.video_path )
        except Exception as e:
            print( 'Yolo_video.create_video_stream(), error: {}'.format( e ) )
            raise

    def load_img( self, current_frame ):
        try:
            img_to_detect = current_frame

            img_height = img_to_detect.shape[0]
            img_width = img_to_detect.shape[1]

            # convert to blob to pass into model
            img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
            # recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
            # accepted sizes are 320×320,416×416,609×609. More size means more accuracy but less speed

            return img_blob

        except Exception as e:
            print( 'Yolo_video.load_img(), error: {}'.format( e ) )
            raise

    def load_model( self ):
        try:
            # Loading pretrained model
            # input preprocessed blob into model and pass through the model
            # obtain the detection predictions by the model using forward() method
            self.yolo_model = cv2.dnn.readNetFromDarknet( self.cfg_path, self.weights_path )

            # Get all layers from the yolo network
            # Loop and find the last layer (output layer) of the yolo network
            yolo_layers = self.yolo_model.getLayerNames()
            self.yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in self.yolo_model.getUnconnectedOutLayers() ]

        except Exception as e:
            print( 'Yolo_video.load_model(), error: {}'.format( e ) )
            raise

    def clean_nms_lists(self):
        try:
            # initialization for non-max suppression (NMS)
            # declare list for [class id], [box center, width & height[], [confidences]
            self.class_ids_list      = []
            self.boxes_list          = []
            self.confidences_list    = []
        except Exception as e:
            print( 'Yolo_video.load_model(), error: {}'.format( e ) )
            raise

    def detect( self, img, img_blob ):
        try:
            img_height = img.shape[0]
            img_width = img.shape[1]

            # input preprocessed blob into model and pass through the model
            self.yolo_model.setInput(img_blob)
            # obtain the detection layers by forwarding through till the output layer
            obj_detection_layers = self.yolo_model.forward( self.yolo_output_layer )

            # loop over each of the layer outputs
            for object_detection_layer in obj_detection_layers:
                # loop over the detections
                for object_detection in object_detection_layer:

                    # obj_detections[1 to 4] => will have the two center points, box width and box height
                    # obj_detections[5] => will have scores for all objects within bounding box
                    all_scores            = object_detection[5:]
                    predicted_class_id    = np.argmax(all_scores)
                    prediction_confidence = all_scores[predicted_class_id]

                    # take only predictions with confidence more than 20%
                    if prediction_confidence > 0.20:
                        # get the predicted label
                        predicted_class_label = self.class_labels[predicted_class_id]
                        # obtain the bounding box co-oridnates for actual image from resized image size
                        bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                        (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                        start_x_pt = int(box_center_x_pt - (box_width / 2))
                        start_y_pt = int(box_center_y_pt - (box_height / 2))

                        ############## NMS Change 2 ###############
                        # save class id, start x, y, width & height, confidences in a list for nms processing
                        # make sure to pass confidence as float and width and height as integers
                        self.class_ids_list.append(predicted_class_id)
                        self.confidences_list.append(float(prediction_confidence))
                        self.boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                        ############## NMS Change 2 END ###########


        except Exception as e:
            print('Yolo_video.detect_img(), error: {}'.format(e))
            raise

    def detect_video( self ):

        i = 0

        #create a while loop
        while ( self.file_video_stream.isOpened):
            i = i + 1
            print( 'frame number: {}'.format( i ) )

            #get the current frame from video stream
            ret,current_frame = self.file_video_stream.read()
            #use the video current frame instead of image

            #img_to_detect = current_frame
            img_blob = self.load_img( current_frame )

            self.clean_nms_lists( )

            self.detect( current_frame, img_blob )

            cv2.imshow("p1", current_frame)

            #terminate while loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        try:
            self.create_video_stream()

            self.load_model()

            self.detect_video()

            #releasing the stream and the camera
            #close all opencv windows
            self.file_video_stream.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print( 'Yolo_video.run(), error: {}'.format( e ) )
            raise


    def __init__( self ):
        try:

            # set of 80 class labels
            self.class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                            "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                            "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                            "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                            "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                            "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

            # Declare List of colors as an array
            # Green, Blue, Red, cyan, yellow, purple
            # Split based on ',' and for every split, change type to int
            # convert that to a numpy array to apply color mask to the image numpy array
            self.class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
            self.class_colors = [np.array(every_color.split(",")).astype("int") for every_color in self.class_colors ]
            self.class_colors = np.array( self.class_colors )
            self.class_colors = np.tile( self.class_colors, (16, 1))

        except Exception as e:
            print( 'Yolo_video.__init__(), error: {}'.format( e ) )
            raise
