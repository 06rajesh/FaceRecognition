import cv2
import os
from operator import itemgetter
import pickle
import sys
import numpy as np
import pandas as pd
import openface
from sklearn.preprocessing import LabelEncoder

# Config URL For MODELS
modelDir = os.path.join('/root/openface/models')
dLibModelDir = os.path.join(modelDir, 'dlib')
openFaceModelDir = os.path.join(modelDir, 'openface')
imgDim = 96


class FaceClassifier:
    def __init__(self, _classifier_dir):
        self.networkModel = os.path.join(openFaceModelDir, 'nn4.small2.v1.t7')
        self.dLibFacePredictor = os.path.join(dLibModelDir, "shape_predictor_68_face_landmarks.dat")
        self.align = openface.AlignDlib(self.dLibFacePredictor)
        self.net = openface.TorchNeuralNet(self.networkModel, imgDim=imgDim, cuda=False)
        with open(_classifier_dir, 'rb') as f:
            if sys.version_info[0] < 3:
                (self.le, self.clf) = pickle.load(f)
            else:
                (self.le, self.clf) = pickle.load(f, encoding='latin1')

    @staticmethod
    def convert_to_arr(_img_path, _convert_to_rgb=True):
        bgr_img = cv2.imread(_img_path)
        if bgr_img is None:
            raise Exception("Unable to load image: {}".format(_img_path))
        if _convert_to_rgb:
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = bgr_img
        return rgb_img

    @staticmethod
    def get_train_data(_data_path):
        f_name = "{}/labels.csv".format(_data_path)
        labels = pd.read_csv(f_name, header=None).as_matrix()[:, 1]
        labels = map(itemgetter(1),
                     map(os.path.split,
                         map(os.path.dirname, labels)))  # Get the Directory
        le = LabelEncoder().fit(labels)
        labels_num = le.transform(labels)
        n_classes = len(le.classes_)
        f_name = "{}/reps.csv".format(_data_path)
        embeddings = pd.read_csv(f_name, header=None).as_matrix()
        return n_classes, labels_num, embeddings

    def get_bounding_boxes(self, _img):
        return self.align.getAllFaceBoundingBoxes(_img)

    def get_rep(self, _img_arr, _get_boxes=False):
        bbs = self.get_bounding_boxes(_img_arr)
        reps = []

        for bb in bbs:
            aligned_face = self.align.align(
                imgDim,
                _img_arr,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                raise Exception("Unable to align image")

            rep = self.net.forward(aligned_face)
            reps.append((bb.center().x, rep))

        if _get_boxes:
            return reps, bbs
        else:
            return reps

    def predict(self, _img_reps):

        # for idx, r in enumerate(_img_reps):
        rep = _img_reps[1].reshape(1, -1)
        predictions = self.clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)

        person = self.le.inverse_transform(maxI)
        confidence = predictions[maxI]

        return person, confidence

