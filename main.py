from math import sqrt
from tensorflow.keras.models import load_model

import cv2
import dlib
import numpy as np

classes = ['bottom', 'front', 'left', 'right', 'top']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
classification_model = load_model("eyes_model.h5")

cap = cv2.VideoCapture(0)
iter = 1
mode = 0

while True:
    _, frame = cap.read()
    c_frame = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(frame, face)

        eye_l = (landmarks.part(36).x, landmarks.part(36).y)
        eye_r = (landmarks.part(39).x, landmarks.part(39).y)
        eye_width = sqrt((eye_r[0] - eye_l[0])**2 + (eye_r[1] - eye_l[1])**2)
        eye_width = int(eye_width * 1.3)
        eye_center_point = (int((eye_r[0] + eye_l[0])/2), int((eye_r[1] + eye_l[1])/2))
        eye_bounding_box = ((int(eye_center_point[0]-eye_width/2),
                             int(eye_center_point[1]-eye_width/2)),
                            (int(eye_center_point[0]+eye_width/2),
                             int(eye_center_point[1]+eye_width/2)))

        crop_x1 = eye_bounding_box[0][0]
        crop_y1 = eye_bounding_box[0][1]
        crop_x2 = eye_bounding_box[1][0]
        crop_y2 = eye_bounding_box[1][1]

        eye_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        eye_img_224 = cv2.resize(eye_region, (224, 224), interpolation=cv2.INTER_AREA)

        class_idx = np.argmax(classification_model.predict(eye_img_224.reshape((1, 224, 224, 3)), verbose=0))

        if mode == 0:

            cv2.rectangle(c_frame, eye_bounding_box[0], eye_bounding_box[1], (0, 255, 0), 1)

            if class_idx == classes.index('front'):
                cv2.circle(c_frame, eye_center_point, 2, (0, 0, 255), 2)
            elif class_idx == classes.index('top'):
                cv2.circle(c_frame, (eye_center_point[0], crop_y1), 2, (0, 0, 255), 2)
            elif class_idx == classes.index('bottom'):
                cv2.circle(c_frame, (eye_center_point[0], crop_y2), 2, (0, 0, 255), 2)
            elif class_idx == classes.index('left'):
                cv2.circle(c_frame, (crop_x2, eye_center_point[1]), 2, (0, 0, 255), 2)
            elif class_idx == classes.index('right'):
                cv2.circle(c_frame, (crop_x1, eye_center_point[1]), 2, (0, 0, 255), 2)

            cv2.imshow("Frame", c_frame)
            print(iter)
            iter += 1
        else:
            cv2.imshow("Eye", eye_img_224)
            print(classes[class_idx])

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
