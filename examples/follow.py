
import face_recognition
import cv2
import time
import queue
import os
from pyparrot.Bebop import Bebop
from DVG import DroneVisionGUI
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import math
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

q = queue.Queue(50)

class Kim():
    def __init__(self, drone_vision):
        self.SIZE = [416, 416]
        self.loop = True
        self.pitch = 0
        self.yaw = 0
        self.vertical = 0
        self.drone_vision = drone_vision
        self.known_face_encodings = []
        self.known_face_names = []
        self.prevscale = -5
        self.prevymin = -5
        self.prevymax = -5
        self.prevxmin = -5
        self.prevxmax = -5
        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True


    def set_p_y_v(self, p, y, v):
        self.pitch = p
        self.yaw = y
        self.vertical = v

    def get_p_y_v(self):
        return self.pitch, self.yaw, self.vertical

    def get_loop(self):
        return self.loop

    def get_frame(self,args):
        # Grab a single frame of video
        frame = self.drone_vision.get_latest_valid_picture()
        if frame is not None:
        # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    min_value = min(distances)

                    # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                    # 0.6 is typical best performance.
                    name = "Unknown"
                    if min_value < 0.5:
                        index = np.argmin(distances)
                        name = self.known_face_names[index]
                        print(self.face_locations)
                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow("result",frame)

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

    def detect_target(self,args):
        #print("pitch : " + str(self.pitch))
        #print("yaw : " + str(self.yaw))
        #print("vertical : " + str(self.vertical))
        category_index = args[0]
        detection_graph = args[1]
        sess = args[2]
        frame = self.drone_vision.get_latest_valid_picture()
        if frame is not None:

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            image_np_expanded = np.expand_dims(rgb_small_frame, axis=0)
            (boxes, scores, classes) = sess.run([boxes, scores, classes],
                        feed_dict={image_tensor: image_np_expanded})
            #print(np.squeeze(boxes)[0])
            #print(self.prevscale)
            flag = ""
            flag,self.prevscale,self.prevymax,self.prevymin,self.prevxmax,self.prevxmin = vis_util.visualize_boxes_and_labels_on_image_array(
                        self.prevymin,
                        self.prevymax,
                        self.prevxmin,
                        self.prevxmax,
                        self.prevscale,
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    # Display output
            cv2.imshow("result", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("dd")
                self.loop = False

            q.put((flag, self.loop))


def tracking_target(droneVision, args):

    print("Press 'q' if you want to stop and land drone")
    loop = True
    drone = args[0]
    status = args[1]
    q = args[2]
    print("ss"+status)
    if status == 't':
        testFlying = True
    else :
        testFlying = False

    if (testFlying):
        drone.safe_takeoff(5)
        drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=1)
    while loop:
        params = q.get()
        flag = params[0]
        if (testFlying):
            if(flag=="goleft"):
                drone.move_relative(0, 0, 0, math.radians(5))
            elif(flag=="goright"):
                drone.move_relative(0, 0, 0, math.radians(-5))
        loop = params[1]
    # land
    if (testFlying):
        drone.safe_land(5)

    # done doing vision demo
    print("Ending the sleep and vision")
    droneVision.close_video()

    drone.smart_sleep(5)

    print("disconnecting")
    drone.disconnect()



if __name__ == "__main__":
    MODEL_NAME = 'mobile'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    # Number of classes to detect
    NUM_CLASSES = 90

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    drone = Bebop()
    success = drone.connect(5)
    #drone.set_picture_format('jpeg')
    is_bebop = True

    if (success):
        # get the state information
        print("sleeping")
        drone.smart_sleep(1)
        drone.ask_for_state_update()
        drone.smart_sleep(1)
        status = input("Input 't' if you want to TAKE OFF or not : ")

        bebopVision = DroneVisionGUI(drone, is_bebop=True, buffer_size=200,user_code_to_run=tracking_target,
                                      user_args=(drone, status, q))
        Kim = Kim(bebopVision)
        bebopVision.set_user_callback_function(Kim.detect_target, user_callback_args=(category_index,detection_graph,sess))
        bebopVision.open_video()

