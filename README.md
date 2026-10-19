# Computer Vision & AI Projects — Combined README

This README describes the supplied Jupyter Notebook projects in one document.

**No images are included in this README or package.** This README is text-only; notebook output images and screenshots have not been copied into the package.

## Projects Included

### YuNet Face + YOLO Object Detection

**Notebook:** `CV_PR3_fixed_clean(2).ipynb`

**Topics found in notebook headings:**
- CV_PR3 — Deep Learning / Computer Vision Practical
- Dataset / files required
- Important execution fix
- Task 1 — Environment Setup, Image Loading & Morphological Operations
- Objective
- Expected output
- Why morphology works
- Task 2 — Bitwise Operations & Image Histograms
- Objective
- Expected output
- Why alpha and beta matter
- Task 3 — Face Detection with YuNet (OpenCV)

### Facial Expression Recognition

**Notebook:** `facial_expression_webcam(1).ipynb`

**Topics found in notebook headings:**
- Facial Expression Recognition – Teachable Machine
- Folder structure

### Animal Voice Detection

**Notebook:** `Animal_Voice_Detection(1).ipynb`

**Topics found in notebook headings:**
- Animal Voice Detection — Teachable Machine Audio Model
- 1. Project files
- 2. Confirm the model metadata
- 3. Start the web app
- 4. Test the classifier

### Pose / Song Recognition

**Notebook:** `pose_song_recognition_updated(1).ipynb`

**Topics found in notebook headings:**
- 🎵 Dance / Song Pose Recognition
- Classes
- Technologies
- 1. Install Required Packages
- 2. Import Libraries
- 3. Project Configuration
- 4. Download MediaPipe Pose Landmarker
- 5. Create MediaPipe Pose Landmarker
- 6. Pose Drawing Utilities
- 7. Feature Extraction
- 8. Webcam Utilities
- 9. Test Webcam + Pose Detection

### ▶️ Video Demonstration

> **Project Explanation Video:**  
> [Click here to watch the complete project explanation](https://drive.google.com/file/d/1yGGVpsVE_sbix7_XuCO8nWq8NGAHi671/view?usp=sharing)


## Setup

Python package requirements differ by notebook. The following common packages support the main computer-vision and notebook workflows. Install additional packages if a notebook reports a missing dependency.

```bash
python -m pip install --upgrade pip
pip install jupyter opencv-contrib-python numpy pandas matplotlib ultralytics
```

## Run the Notebooks

1. Extract the ZIP file.
2. Open a terminal in the project folder.
3. Run `jupyter notebook`.
4. Open the required `.ipynb` file.
5. Run the notebook cells in order and follow any dataset/model setup instructions in that notebook.

Webcam projects require camera access. Some models or resources may be downloaded when their notebook cells run.

## Important Notes

- The README is text-only and contains no embedded images or image links.
- Keep the original notebook files alongside this README when submitting or uploading the project.
- Dependencies may differ between projects; follow each notebook's setup instructions if an additional package is required.

## Author

Update with your name, college, department, semester, and project submission details.
