from math import sqrt

import os
import cv2
import dlib
import random
import string

if os.path.isdir("data") is False:
    os.mkdir('data')

classes = ['bottom', 'front', 'left', 'right', 'top']
for im_class in classes:
    if os.path.isdir(f"data/{im_class}") is False:
        os.mkdir(f'data/{im_class}')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

for i in range(100):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        # cv2.rectangle(frame, eye_bounding_box[0], eye_bounding_box[1], (0, 255, 0), 1)

        crop_x1 = eye_bounding_box[0][0]
        crop_y1 = eye_bounding_box[0][1]
        crop_x2 = eye_bounding_box[1][0]
        crop_y2 = eye_bounding_box[1][1]

        eye_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        eye_img_224 = cv2.resize(eye_region, (224, 224), interpolation=cv2.INTER_AREA)

        # cv2.imshow("Frame", frame)
        cv2.imshow("Eye", eye_img_224)

        ##### cv2.imwrite(f'data/front/eye_{"".join(random.choices(string.ascii_lowercase, k=15))}.jpg', eye_img_224)
        ##### cv2.imwrite(f'data/bottom/eye_{"".join(random.choices(string.ascii_lowercase, k=15))}.jpg', eye_img_224)
        ##### cv2.imwrite(f'data/left/eye_{"".join(random.choices(string.ascii_lowercase, k=15))}.jpg', eye_img_224)
        ##### cv2.imwrite(f'data/right/eye_{"".join(random.choices(string.ascii_lowercase, k=15))}.jpg', eye_img_224)
        ##### cv2.imwrite(f'data/top/eye_{"".join(random.choices(string.ascii_lowercase, k=15))}.jpg', eye_img_224)
    cv2.waitKey(1)

cap.release()