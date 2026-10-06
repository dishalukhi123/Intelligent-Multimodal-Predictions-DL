"""
Pose Song Recognition - Python only

5 classes:
1. Chal Chaiya Chaiya
2. You Are My Soniya
3. Bole Chudiyan
4. Chammak Challo
5. Kajra Re

No HTML and no tensorflowjs.
Uses MediaPipe Tasks API + OpenCV + Random Forest.

Run:
    python pose_song_recognition.py
"""

import sys
import subprocess
import urllib.request
from pathlib import Path

REQUIRED = ["mediapipe", "opencv-python", "numpy", "scikit-learn", "joblib"]


def install_packages():
    """Install packages in the current Python environment if missing."""
    import importlib.util
    for package in REQUIRED:
        module = package.replace("-", "_")
        if importlib.util.find_spec(module) is None:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])


install_packages()

import cv2
import numpy as np
import mediapipe as mp
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = [
    "Chal Chaiya Chaiya",
    "You Are My Soniya",
    "Bole Chudiyan",
    "Chammak Challo",
    "Kajra Re",
]

DATA_FILE = Path("pose_training_data.npz")
MODEL_FILE = Path("pose_song_classifier.joblib")
POSE_MODEL_FILE = Path("pose_landmarker_full.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# -------------------- MediaPipe Tasks --------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def download_pose_model():
    if POSE_MODEL_FILE.exists() and POSE_MODEL_FILE.stat().st_size > 100000:
        print(f"Pose model already exists: {POSE_MODEL_FILE}")
        return
    print("Downloading MediaPipe Pose Landmarker model...")
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_FILE)
    except Exception as exc:
        if POSE_MODEL_FILE.exists():
            POSE_MODEL_FILE.unlink()
        raise RuntimeError(
            "Could not download the MediaPipe model. Check your internet connection."
        ) from exc
    print("Pose model downloaded successfully.")


def create_pose_landmarker():
    download_pose_model()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_FILE)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


# -------------------- Drawing --------------------
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


def draw_pose(frame, result):
    if not result.pose_landmarks:
        return
    landmarks = result.pose_landmarks[0]
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        points.append((
            int(np.clip(lm.x, 0.0, 1.0) * w),
            int(np.clip(lm.y, 0.0, 1.0) * h),
        ))
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


# -------------------- Features --------------------
def extract_pose_features(result):
    """33 landmarks x (x,y,z,visibility) = 132 normalized features."""
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    points = []
    for lm in landmarks:
        visibility = getattr(lm, "visibility", 1.0)
        points.append([lm.x, lm.y, lm.z, visibility])

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] != 33:
        return None

    hip_center = (points[23, :3] + points[24, :3]) / 2.0
    points[:, :3] -= hip_center

    scale = np.linalg.norm(points[11, :3] - points[12, :3])
    if scale < 1e-6:
        scale = np.linalg.norm(points[23, :3] - points[24, :3])
    if scale < 1e-6:
        scale = 1.0

    points[:, :3] /= scale
    return points.flatten().astype(np.float32)


# -------------------- Camera --------------------
def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or try camera index 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def detect_pose(pose_landmarker, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return pose_landmarker.detect(mp_image)


# -------------------- Webcam test --------------------
def webcam_test():
    print("Starting webcam test. Press Q to stop.")
    pose_landmarker = create_pose_landmarker()
    cap = open_camera()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = detect_pose(pose_landmarker, frame)
            draw_pose(frame, result)
            text = "POSE DETECTED" if result.pose_landmarks else "NO POSE"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Pose Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_landmarker.close()


# -------------------- Collect training data --------------------
def collect_training_data():
    print("\n" + "=" * 60)
    print("WEBCAM TRAINING DATA COLLECTION")
    print("=" * 60)
    for i, label in enumerate(LABELS, 1):
        print(f"{i} -> {label}")
    print("\nPress 1-5 to select a class.")
    print("Move/dance naturally to collect different poses.")
    print("Collect at least 100-200 samples per class.")
    print("Press Q to finish.")

    pose_landmarker = create_pose_landmarker()
    cap = open_camera()
    all_features, all_labels = [], []
    current_label = None
    counts = {i: 0 for i in range(len(LABELS))}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = detect_pose(pose_landmarker, frame)
            draw_pose(frame, result)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
                current_label = int(chr(key)) - 1
                print(f"Selected: {LABELS[current_label]}")
            if key == ord("q"):
                break

            if current_label is not None and result.pose_landmarks:
                features = extract_pose_features(result)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(current_label)
                    counts[current_label] += 1

            cv2.rectangle(frame, (5, 5), (635, 145), (0, 0, 0), -1)
            selected = (
                "No class selected - press 1 to 5"
                if current_label is None
                else f"Class {current_label + 1}: {LABELS[current_label]}"
            )
            current_count = counts[current_label] if current_label is not None else 0
            cv2.putText(frame, selected, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Total samples: {len(all_features)}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Current class samples: {current_count}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "1-5 Select Class | Q Quit", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Pose Training", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_landmarker.close()

    if not all_features:
        print("No samples were collected.")
        return False

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int32)
    np.savez(DATA_FILE, X=X, y=y, labels=np.asarray(LABELS))

    print("\nTraining data saved:", DATA_FILE.absolute())
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    for i, label in enumerate(LABELS):
        print(f"{i + 1}. {label}: {np.sum(y == i)} samples")
    return True


# -------------------- Dataset --------------------
def load_training_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Collect training data first.")
    data = np.load(DATA_FILE, allow_pickle=True)
    return data["X"], data["y"]


def show_dataset():
    try:
        X, y = load_training_data()
    except FileNotFoundError as exc:
        print(exc)
        return
    print("\nDataset shape:", X.shape)
    print("Labels shape:", y.shape)
    for i, label in enumerate(LABELS):
        print(f"{i + 1}. {label}: {np.sum(y == i)} samples")


# -------------------- Train model --------------------
def train_model():
    try:
        X, y = load_training_data()
    except FileNotFoundError as exc:
        print(exc)
        return False

    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        print("At least 2 classes are required.")
        return False
    if np.min(counts) < 2:
        print("Every class needs at least 2 samples.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print("\nModel saved:", MODEL_FILE.absolute())
    return True


# -------------------- Live prediction --------------------
def live_prediction():
    if not MODEL_FILE.exists():
        print(f"{MODEL_FILE} does not exist. Train the model first.")
        return

    model = joblib.load(MODEL_FILE)
    pose_landmarker = create_pose_landmarker()
    cap = open_camera()

    print("\nLIVE SONG PREDICTION - Press Q to stop")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = detect_pose(pose_landmarker, frame)
            draw_pose(frame, result)

            prediction_text = "No pose detected"
            confidence_text = ""

            if result.pose_landmarks:
                features = extract_pose_features(result)
                if features is not None:
                    probabilities = model.predict_proba(features.reshape(1, -1))[0]
                    prediction_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[prediction_index])
                    prediction_text = LABELS[prediction_index]
                    confidence_text = f"Confidence: {confidence * 100:.1f}%"

            cv2.rectangle(frame, (5, 5), (635, 115), (0, 0, 0), -1)
            cv2.putText(frame, prediction_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Dance Song Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_landmarker.close()


# -------------------- Menu --------------------
def main():
    print("\n" + "=" * 60)
    print("       DANCE / SONG POSE RECOGNITION")
    print("=" * 60)
    print("1. Test Webcam + Pose")
    print("2. Collect Training Data")
    print("3. Show Dataset")
    print("4. Train Model")
    print("5. Live Song Prediction")
    print("6. Collect Data + Train Model")
    print("0. Exit")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        webcam_test()
    elif choice == "2":
        collect_training_data()
    elif choice == "3":
        show_dataset()
    elif choice == "4":
        train_model()
    elif choice == "5":
        live_prediction()
    elif choice == "6":
        if collect_training_data():
            train_model()
    elif choice == "0":
        print("Goodbye.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
