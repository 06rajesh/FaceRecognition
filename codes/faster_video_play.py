# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import numpy as np
import cv2
import time
from FaceClassifier import FaceClassifier

if __name__ == '__main__':
    clf = FaceClassifier('/media/embeddings/myClassifier.pkl')
    fvs = FileVideoStream('shakib.mp4').start()
    time.sleep(1.0)
    fps = FPS().start()

    while fvs.more():

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = fvs.read()
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])

        reps, bbs = clf.get_rep(frame, True)

        for idx, bb in enumerate(bbs):
            person, confidence = clf.predict(reps[idx])
            cv2.rectangle(frame, (bbs[idx].left(), bbs[idx].top()), (bbs[idx].right(), bbs[idx].bottom()),
                          (0, 255, 0), 2)
            if confidence > 0.75:
                cv2.putText(frame, "{} @{:.2f}".format(person.decode('utf-8'), confidence),
                            (bbs[idx].left(), bbs[idx].bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    fvs.stop()
    cv2.destroyAllWindows()
