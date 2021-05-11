# Import the required packges
import cv2 as cv
import numpy as np
import os
from scipy.spatial import distance as dist
import imutils

# Set the minimum confidence and nms thresholds
confidence_thresh = 0.5
nms_thresh = 0.2

# Set the minimum appropriate distance in pixels
min_distance = 50

# Load the path of the cofig and weights file for Yolov3
weights_path = os.path.sep.join(['config and weights/yolov3.weights'])
config_path = os.path.sep.join(['config and weights/yolov3.cfg'])

# Load the names file
names_path = 'config and weights/coco.names'
names = []
with open(names_path, 'rt') as f:
    names = f.read().rstrip('\n').split('\n')

# Creating/Loading our Yolov3 model
net = cv.dnn.readNetFromDarknet(config_path, weights_path)

# Set OpenCV as backend and Specify the use of CPU
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get layer names where the actual predictions are made
layer_names = net.getLayerNames()
output_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# Read the video stream
capture = cv.VideoCapture('example.mp4')
vid_writer = None


def detection(frame, net, layer_names, id_person = 0, target_size = 416):
    hT, wT, cT = frame.shape
    bounding_box = []
    class_ids = []
    confidence_list = []
    centroids = []
    results = []

    # Convert the frame to blob so that the network understand it when we formard pass it.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (target_size, target_size), [0, 0, 0], 1, crop=False)

    # Set this blob as input to our network
    net.setInput(blob)

    outputs = net.forward(layer_names)

    for output in outputs:
        for detect in output:
            # Exclude the first 5 values to get the confidence score
            scores = detect[5:]

            # Get the class id with the highest confidence score
            class_ID = np.argmax(scores)

            # Save the confidence of the class with the highest confidence score
            conf = scores[class_ID]

            # Store the corresponding values of said class
            if conf > confidence_thresh and class_ID == id_person:
                # Get the values required for creating a bounding box and also get the centriod values
                W, H = int(detect[2]*wT), int(detect[3]*hT)
                X, Y = int((detect[0] * wT) - W / 2), int((detect[1] * hT) - H / 2)

                # Save the appropriate values
                bounding_box.append([X, Y, W, H])
                class_ids.append(class_ID)
                confidence_list.append(float(conf))
                centroids.append((int(detect[0] * wT), int(detect[1] * hT)))

    # Apply NMS thresholding
    # Non Maxima Thresholding is done to remove unwanted, weak bounding boxes
    indices = cv.dnn.NMSBoxes(bounding_box, confidence_list, confidence_thresh, nms_thresh)

    # Iterate over the indices
    for i in indices:
        i = i[0]
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        r = (confidence_list[i], (x, y, x + w, y + h), centroids[i])
        results.append(r)

    return results


while True:
    is_true, frame = capture.read()

    if not is_true:
        break

    frame = imutils.resize(frame, width=700)
    result = detection(frame, net, output_names, id_person = names.index('person'))

    centroids = np.array([r[2] for r in result])
    distance = []

    social_distancing_violation = set()

    for i in centroids:
        a = []
        for j in centroids:
            a.append(np.linalg.norm(i - j))
        distance.append(a)

    for i in range(0, len(distance)):
        for j in range(i+1, len(distance[i])):
            if distance[i][j] < min_distance:
                social_distancing_violation.add(i)
                social_distancing_violation.add(j)

    for (i, (conf, bounding_box, centroid)) in enumerate(result):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bounding_box
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then
        # update the color
        if i in social_distancing_violation:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "{0} out of {1} people are violating social distancing". format(len(social_distancing_violation), len(result))
    cv.putText(frame, text, (10, frame.shape[0] - 25), cv.FONT_HERSHEY_DUPLEX, 0.82, (0, 0, 250), 2)

    cv.imshow('Social Distancing using YOLOv3', frame)

    if cv.waitKey(20) & 0xff == ord('d'):
        break

    if vid_writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        vid_writer = cv.VideoWriter('output.mp4', fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    else:
        vid_writer.write(frame)

    cv.waitKey(1)
