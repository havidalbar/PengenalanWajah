# import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

# load serialized face detector
print("Loading Face Detector...")
protoPath = "model/deploy.prototxt"
modelPath = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# hbd = cv2.imread('hbd.png', -1)


# func to merge image background and foreground
# def paste(back, x, y):
#     x = x - 20
#     fore = cv2.imread('hbd.png', cv2.IMREAD_UNCHANGED)
#     fore = cv2.resize(fore,(250,250))
#     y = y - 250
#     if back.shape[2] == 3:
#         back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
#     if fore.shape[2] == 3:
#         fore = cv2.cvtColor(fore, cv2.COLOR_BGR2BGRA)

#     rows, cols, channels = fore.shape    
#     trans_indices = fore[...,3] != 0 # Where not transparent
#     overlay_copy = back[y:y+rows, x:x+cols] 
#     overlay_copy[trans_indices] = fore[trans_indices]
#     back[y:y+rows, x:x+cols] = overlay_copy
#     return cv2.cvtColor(back, cv2.COLOR_BGRA2BGR)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=1000)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			# color = (255, 0, 0)
			color = (0,255,0)
			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			#call the func merge image
			# frame = paste(frame, startX, startY)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				color, 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			#convert frame result 4chanel to 3chanel
			# frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()