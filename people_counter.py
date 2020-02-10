# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input test.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary package

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from collections import deque

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=2,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

# red --> blue exit
# blue --> red enter

# define red rect
x1_r = 20
y1_r = 130
x2_r = 490
y2_r = 280
# define blue rect
x1_b = 20
y1_b = 0
x2_b = 486
y2_b = 128


def is_in_rect_red(x, y):
    """ Check if the centroid of the people in red rectangle """
    if x1_r <= x <= x2_r and y1_r <= y <= y2_r:
        return True
    else:
        return False


def is_in_rect_blue(x, y):
    """ Check if the centroid of the people in blue rectangle """
    if x1_b <= x <= x2_b and y1_b <= y <= y2_b:
        return True
    else:
        return False


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=2).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=100)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0  # using B --> R
totalDown_Motion = 0  # number of people counted using motion algorithm
totalUp = 0  # using R --> B
totalUp_Motion = 0  # number of people counted using motion algorithm
max_confidence_in_red = 0
max_confidence_in_blue = 0
total_visitors = 0


# start the frames per second throughput estimator
fps = FPS().start()
is_in_red = False
is_in_blue = False

# for motion detection
red_frame_list = deque()
blue_frame_list = deque()


def have_motion(frame_no, frame_list):
    """ Check if there is any motion on the opposite rectangle """
    # initialize the first frame in the video stream
    firstFrame = None
    for x in frame_list:
        frame = x
        if frame is not None:
            # print('frame is not none')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if firstFrame is None:
                firstFrame = gray
                continue
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            # cv2.imwrite(f"output_frames/frame{frame_no}.jpg", thresh)
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < 5000:
                    continue
                elif len(cnts) >= 1:
                    # save the resultant image for debugging purpose
                    cv2.imwrite(f"output_frames/frame{frame_no}.jpg", thresh)
                    # print(f"length of contour: {len(cnts)}")
                    # if counting is done, then clear the respective frame buffer
                    frame_list.clear()
                    return True
            frame_no += 1


# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream

    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=512, height=512)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers

    # status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        # set the status and initialize our new set of object trackers
        # status = "Detecting"
        trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    # print(f"Another object: {CLASSES[idx]}")
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames 
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            # status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        # print(len(objects.items()))

        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction

        else:
            # all (x,y) co-ordinates of centroids
            # y = [c[1] for c in to.centroids]
            # x = [c[0] for c in to.centroids]

            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:

                if is_in_rect_blue(centroid[0], centroid[1]):

                    is_in_blue = True

                    for c in to.centroids:
                        if is_in_rect_red(c[0], c[1]):
                            print("R --> B")
                            totalUp += 1
                            to.counted = True
                            is_in_blue = False
                            is_in_red = False
                            break

                    if not to.counted and have_motion(totalFrames, red_frame_list) and not is_in_red:
                        print("Entered (motion detector)")
                        totalDown_Motion += 1
                        is_in_red = False
                        is_in_blue = False
                        to.counted = True

                elif is_in_rect_red(centroid[0], centroid[1]):
                    # print(f"Red: {objectID}")
                    is_in_red = True

                    for c in to.centroids:
                        if is_in_rect_blue(c[0], c[1]):
                            print("B --> R")
                            totalDown += 1
                            to.counted = True
                            is_in_red = False
                            is_in_blue = False
                            break

                    if not to.counted and have_motion(totalFrames, blue_frame_list) and not is_in_blue:
                        print("Exited (motion detector)")
                        totalUp_Motion += 1
                        is_in_red = False
                        is_in_blue =False
                        to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        # print(f"red: {is_in_red}, blue: {is_in_blue}, id: {objectID}")
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 20, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    total_entered = (totalDown * 0.95) + (totalDown_Motion * 0.85)  # weighted avg.
    total_exited = (totalUp * 0.95) + (totalUp_Motion * 0.85)  # weighted avg.

    total_visitors = round((total_entered + total_exited)/2)

    info = [
        ("Entered ", total_entered),
        ("Exited ", total_exited),
        ("Total Visitors: ", total_visitors)
    ]
    # just beautify the vision, nothing else
    if is_in_red:
        cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 0), 2)  # make it green when any person in red rect
        cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), (255, 0, 0), 2)  # blue (outside)
    elif is_in_blue:
        cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)  # red (inside)
        cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)  # make it green when any person in blue rect
    else:
        cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)  # red (inside)
        cv2.rectangle(frame, (x1_b, y1_b), (x2_b, y2_b), (255, 0, 0), 2)  # blue (outside)

    # save 100 future frames for contour detection and count
    # print(is_in_blue)
    if is_in_red:
        blue_rect_crop = frame[y1_b:y2_b, x1_b:x2_b]
        if len(blue_frame_list) >= 100:
            blue_frame_list.popleft()
        # cv2.imwrite(f"output_frames/frame_blue{totalFrames}.jpg", blue_rect_crop)
        blue_frame_list.append(blue_rect_crop)

    elif is_in_blue:
        red_rect_crop = frame[y1_r:y2_r, x1_r:x2_r]
        if len(red_frame_list) >= 100:
            red_frame_list.popleft()
        # cv2.imwrite(f"output_frames/frame_red{totalFrames}.jpg", red_rect_crop)
        red_frame_list.append(red_rect_crop)

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (50, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
