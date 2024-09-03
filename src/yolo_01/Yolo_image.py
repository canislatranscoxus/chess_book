# -*- coding: utf-8 -*-
"""

@author: Canis Latrans Coxus
"""

import numpy as  np
import cv2

class Yolo_image:


    # file paths
    image_path           = '/home/art/Pictures/yolo/scene2.jpg'
    img_to_detect        = None

    # convert to blob to pass into model
    img_blob             = None

    class_labels         = None
    class_colors         = None

    prediction_threshold = 0.20
    yolo_model           = None
    yolo_output_layer    = None

    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = None

    def load_img(self):
        try:
            # load the image to detect, get width, height
            self.img_to_detect = cv2.imread( self.image_path )

            # convert to blob to pass into model
            self.img_blob = cv2.dnn.blobFromImage( self.img_to_detect,
                                             scalefactor=1.0 / 255.0,
                                             size=(416, 416),
                                             swapRB=True,
                                             crop=False)
            # recommended by yolo authors, scale factor is 0.003922=1/255, width,height of blob is 320,320
            # accepted sizes are 320×320,416×416,609×609. More size means more accuracy but less speed

        except Exception as e:
            print( 'Yolo_image.load_img(), error: {}'.format( e ) )
            raise

    def load_model(self):
        try:
            # Loading pretrained model
            # input preprocessed blob into model and pass through the model
            # obtain the detection predictions by the model using forward() method
            self.yolo_model = cv2.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')

            # Get all layers from the yolo network
            # Loop and find the last layer (output layer) of the yolo network
            yolo_layers = self.yolo_model.getLayerNames()
            self.yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in self.yolo_model.getUnconnectedOutLayers()]

            # input preprocessed blob into model and pass through the model
            self.yolo_model.setInput( self.img_blob )

        except Exception as e:
            print('Yolo_image.load_model(), error: {}'.format(e))
            raise

    def draw_boxes(self, img_to_detect, object_detection):
        try:
            img_height = img_to_detect.shape[0]
            img_width  = img_to_detect.shape[1]

            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            # take only predictions with confidence more than 20%
            if prediction_confidence > self.prediction_threshold:
                #get the predicted label
                predicted_class_label = self.class_labels[predicted_class_id]
                #obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                end_x_pt = start_x_pt + box_width
                end_y_pt = start_y_pt + box_height

                #get a random mask color from the numpy array of colors
                box_color = self.class_colors[predicted_class_id]

                #convert the color numpy array as a list and apply to text and box
                box_color = [int(c) for c in box_color]

                # print the prediction in console
                predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                print("predicted object {}".format(predicted_class_label))

                # draw rectangle and text in the image
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            return img_to_detect
        except Exception as e:
            print( 'Yolo_image.draw_boxes(), error: {}'.format( e ) )
            raise


    def detect(self):
        try:
            # obtain the detection layers by forwarding through till the output layer
            obj_detection_layers = self.yolo_model.forward( self.yolo_output_layer )

            # loop over each of the layer outputs
            for object_detection_layer in obj_detection_layers:
                # loop over the detections
                for object_detection in object_detection_layer:

                    self.draw_boxes( self.img_to_detect, object_detection )

        except Exception as e:
            print('Yolo_image.detect(), error: {}'.format(e))
            raise

    def show_results(self, img ):
        try:
            cv2.imshow("Detection Output", img )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print('Yolo_image.show_results(), error: {}'.format(e))
            raise


    def run(self):
        try:
            print( 'run() ... begin' )

            self.load_img()
            self.load_model()
            self.detect()
            self.show_results( self.img_to_detect )
            print( 'run() ... end' )
        except Exception as e:
            print('Yolo_image.run(), error: {}'.format(e))
            raise


    def __init__(self):
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
            self.class_colors = [np.array(every_color.split(",")).astype("int") for every_color in self.class_colors]
            self.class_colors = np.array( self.class_colors )
            self.class_colors = np.tile( self.class_colors, (16, 1))

        except Exception as e:
            print( 'Yolo_image.__init__(), error: {}'.format(e) )
            raise

if __name__ == '__main__':
    yolo = Yolo_image()
    yolo.run()