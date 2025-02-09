import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Load the face detection model
faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Predict if the person is wearing a mask or not
            face = np.expand_dims(face, axis=0)
            mask, withoutMask = maskNet.predict(face)[0]

            # Determine the label (Mask/No Mask)
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display the label and rectangles
            label = f'{label}: {max(mask, withoutMask) * 100:.2f}%'
            cv2.putText(frame, label, (startX, endY + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw rectangles around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
