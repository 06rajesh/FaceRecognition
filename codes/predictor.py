import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/root/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
classifierDir = os.path.join('/media/embeddings/myClassifier.pkl')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getImageArray(img_path, convert_to_rgb=True):
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise Exception("Unable to load image: {}".format(img_path))
    if convert_to_rgb:
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = bgr_img
    return rgb_img


def get_bounding_boxes(_img):
    return align.getAllFaceBoundingBoxes(_img)


def getRep(imgPath):
    start = time.time()
    rgbImg = getImageArray(imgPath)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()
    bbs = get_bounding_boxes(rgbImg)
    if len(bbs) == 0:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if args.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        if args.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        reps.append((bb.center().x, rep))
    # sreps = sorted(reps, key=lambda x: x[0])
    return reps, bbs


def infer(args):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')

    for img in args.imgs:
        imgArr = getImageArray(img, False)
        print("\n=== {} ===".format(img))
        reps, bbs = getRep(img)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for idx, r in enumerate(reps):
            print(r)
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            if args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            else:
                if confidence > 0.55:
                    cv2.putText(imgArr, "{} @{:.2f}".format(person.decode('utf-8'), confidence),
                                (bbs[idx].left(), bbs[idx].bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)
                    cv2.rectangle(imgArr, (bbs[idx].left(), bbs[idx].top()), (bbs[idx].right(), bbs[idx].bottom()),
                                  (0, 255, 0), 2)
                cv2.imshow('', imgArr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument(
        '--classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT '
             'the Torch network model, which can be set with --networkModel.',
             default=classifierDir)
    parser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))
    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()
    infer(args)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
