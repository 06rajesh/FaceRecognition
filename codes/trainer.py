#!/usr/bin/env python2
#
# Trainer to classify faces.
# Rajesh Baidya
# 2018/05/13
#

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/root/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def get_image_array(_img_path, _convert_to_rgb=True):
    bgr_img = cv2.imread(_img_path)
    if bgr_img is None:
        raise Exception("Unable to load image: {}".format(_img_path))
    if _convert_to_rgb:
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = bgr_img
    return rgb_img


def get_rep(_img_path):
    rgb_img = get_image_array(_img_path)
    bbs = align.getAllFaceBoundingBoxes(rgb_img)
    if len(bbs) == 0:
        raise Exception("Unable to find a face: {}".format(_img_path))
    reps = []
    for bb in bbs:
        aligned_face = align.align(
            args.imgDim,
            rgb_img,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align image: {}".format(_img_path))

        rep = net.forward(aligned_face)
        reps.append((bb.center().x, rep))

    return reps, bbs


def train(_args):
    print("Loading embeddings.")
    f_name = "{}/labels.csv".format(_args.workDir)
    labels = pd.read_csv(f_name, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the Directory
    le = LabelEncoder().fit(labels)
    labels_num = le.transform(labels)
    n_classes = len(le.classes_)
    f_name = "{}/reps.csv".format(_args.workDir)
    embeddings = pd.read_csv(f_name, header=None).as_matrix()
    print("Training for {} classes.".format(n_classes))
    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, labels_num)

    f_name = "{}/myClassifier.pkl".format(args.workDir)
    print("Saving classifier to '{}'".format(f_name))
    with open(f_name, 'w') as f:
        pickle.dump((le, clf), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dlibFacePredictor', type=str,
        help='Path to dlib Face Predictor',
        default=os.path.join(dlibModelDir, 'shape_predictor_68_face_landmarks.dat'))

    parser.add_argument(
        '--networkModel', type=str,
        help="Path to Torch network model.",
        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int, help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')

    parser.add_argument(
        'workDir', type=str,
        help="The input work directory containing 'reps.csv' and 'labels.csv'. "
             "Obtained from aligning a directory with 'align-dlib' and getting the"
             " representations with 'batch-represent'.")

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)
    train(args)
