"""
Real-Time Facial Expression Recognition
Teachable Machine + Keras + OpenCV

Required files in the same folder:
    keras_Model.h5
    labels.txt
    facial_expression.py

Install:
    python -m pip install tensorflow keras opencv-python numpy

Run:
    python facial_expression.py

Press Q or ESC to quit.
"""

import os
import sys
import cv2
import numpy as np

# Try Keras first; TensorFlow's Keras is also supported.
try:
    from keras.models import load_model
except ModuleNotFoundError:
    try:
        from tensorflow.keras.models import load_model
    except ModuleNotFoundError:
        print("ERROR: Keras/TensorFlow is not installed.")
        print("Run:")
        print("python -m pip install tensorflow keras")
        sys.exit(1)


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
CAMERA_INDEX = 0

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


# -----------------------------
# Check files
# -----------------------------
if not os.path.isfile(MODEL_PATH):
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    print("Put keras_Model.h5 in the same folder as this Python file.")
    sys.exit(1)

if not os.path.isfile(LABELS_PATH):
    print(f"ERROR: Labels file not found: {LABELS_PATH}")
    print("Put labels.txt in the same folder as this Python file.")
    sys.exit(1)


# -----------------------------
# Load model
# -----------------------------
print("Loading model...")

try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    print("ERROR: Could not load the model.")
    print(e)
    sys.exit(1)

print("Model loaded successfully.")


# -----------------------------
# Load labels
# -----------------------------
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file if line.strip()]
except Exception as e:
    print("ERROR: Could not read labels.txt")
    print(e)
    sys.exit(1)

if not class_names:
    print("ERROR: labels.txt is empty.")
    sys.exit(1)

print("Classes:", class_names)


# -----------------------------
# Start webcam
# -----------------------------
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    print("Try changing CAMERA_INDEX = 0 to CAMERA_INDEX = 1.")
    sys.exit(1)

print("\nCamera started.")
print("Show your face to the camera.")
print("Press Q or ESC to quit.\n")


# -----------------------------
# Real-time prediction loop
# -----------------------------
while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read frame from webcam.")
        break

    # Mirror the webcam image so it behaves naturally.
    frame = cv2.flip(frame, 1)

    # Keep a copy for display.
    display_frame = frame.copy()

    # Resize image to the Teachable Machine input size.
    resized = cv2.resize(
        frame,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Convert image to float32 and reshape.
    image = np.asarray(resized, dtype=np.float32)
    image = image.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    # Teachable Machine normalization.
    image = (image / 127.5) - 1.0

    # Predict.
    prediction = model.predict(image, verbose=0)

    # Find the class with the highest probability.
    index = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][index])

    # Make sure the index exists in labels.txt.
    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = f"Class {index}"

    confidence_percent = confidence * 100

    # Remove a leading numeric label such as "0 " or "1 ".
    if len(class_name) > 2 and class_name[0].isdigit() and class_name[1].isspace():
        class_name = class_name[2:].strip()

    # Print prediction in terminal.
    print(
        f"Expression: {class_name} | "
        f"Confidence: {confidence_percent:.1f}%",
        end="\r"
    )

    # -----------------------------
    # Display result on webcam
    # -----------------------------
    cv2.rectangle(
        display_frame,
        (10, 10),
        (520, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        f"Expression: {class_name}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        f"Confidence: {confidence_percent:.1f}%",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        "Press Q or ESC to quit",
        (20, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Facial Expression Recognition", display_frame)

    # Q or ESC closes the application.
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# -----------------------------
# Cleanup
# -----------------------------
camera.release()
cv2.destroyAllWindows()

print("\n\nCamera stopped.")
print("Program finished.")
