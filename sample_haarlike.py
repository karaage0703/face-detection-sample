#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse
import urllib.request
import os

import cv2 as cv


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--face", action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    url_model = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml'
    path_model = './model/haarcascade_frontalface_alt.xml'

    url_face_image = 'https://raw.githubusercontent.com/karaage0703/karaage_icon/master/karaage_icon.png'
    path_face_image = './model/karaage_icon.png'


    is_file = os.path.isfile(path_model)
    if not is_file:
        print('model file downloading...')
        urllib.request.urlretrieve(url_model, path_model)

    is_file = os.path.isfile(path_face_image)
    if not is_file:
        print('face image file downloading...')
        urllib.request.urlretrieve(url_face_image, path_face_image)


    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        cascade = cv.CascadeClassifier(path_model)
        gray_image = cv.cvtColor(debug_image, cv.COLOR_BGR2GRAY)

        face = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        if args.face is True:
            debug_image = face_overlay(debug_image, face, path_face_image)
        else:
            for (x, y, w, h) in face:
                cv.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255,0), 2)

        elapsed_time = time.time() - start_time

        text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
        text = text + 'ms'
        debug_image = cv.putText(
            debug_image,
            text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('Open CV haarlike Sample', debug_image)

    cap.release()
    cv.destroyAllWindows()


def face_overlay(image, face, path_face_image):
    # image padding
    padding_size = int(image.shape[1] / 2)
    padding_img = cv.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = cv.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = image_tmp.astype('float64')

    # face overlay
    if len(face) > 0:
        for rect in face:
            face_size = rect[2] * 2
            face_pos_adjust = int(rect[2] * 0.5)
            face_img = cv.imread(path_face_image, cv.IMREAD_UNCHANGED)
            face_img = cv.resize(face_img, (face_size, face_size))
            mask = face_img[:,:,3]
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            mask = mask / 255.0
            face_img = face_img[:,:,:3]

            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] *= 1 - mask
            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] += face_img * mask

    image_tmp = image_tmp[padding_size:padding_size+image.shape[0], padding_size:padding_size+image.shape[1]]
    image_tmp = image_tmp.astype('uint8')

    return image_tmp


if __name__ == '__main__':
    main()
