# import the necessary packages
from imutils.video import FPS
import imutils
import numpy as np
import cv2
import argparse
import os
import openface
# import predictor as pred

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/root/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')


def get_bounding_boxes(_img):
    return align.getAllFaceBoundingBoxes(_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))

    args = parser.parse_args()
    align = openface.AlignDlib(args.dlibFacePredictor)

    stream = cv2.VideoCapture('shakib.mp4')
    fps = FPS().start()

    while True:
        grabbed, frame = stream.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # resize the frame and convert it to grayscale (while still
        # retaining 3 channels)
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

        bbs = get_bounding_boxes(frame)

        for bb in bbs:
            cv2.rectangle(frame, (bb.left(), bb.top()), (bb.right(), bb.bottom()),
                          (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    stream.release()
    cv2.destroyAllWindows()
