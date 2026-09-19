# Deteksi Atribut Seragam Sekolah (School Uniform Attribute Detection)

This project is a computer vision application that detects school uniform attributes (Dasi/Tie, Logo Sekolah/School Badge, Topi/Hat, and Ikat Pinggang/Belt) in real-time using a webcam feed. It utilizes Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier to identify the objects through a sliding window approach.

## Features
- Detects the following attributes:
  - Dasi (Tie)
  - Logo Sekolah (School Badge)
  - Topi (Hat)
  - Ikat Pinggang (Belt)
- Real-time webcam integration.
- Facial region exclusion to reduce false positives (using OpenCV Haar Cascades).
- Custom Non-Maximum Suppression (NMS) for refining overlapping bounding boxes.

## How It Works
1. **Feature extraction** — each object crop is resized to 96x96 and turned into a HOG (Histogram of Oriented Gradients) feature vector, which captures shape/edge patterns.
2. **Training** — an SVM classifier (linear kernel) is trained on these HOG features, using both the annotated objects and randomly sampled background patches as negative examples. Simple data augmentation (rotation, blur, noise) is applied to boost training data.
3. **Detection** — the webcam frame is scanned with a sliding window; each window is classified by the SVM, filtered by a confidence threshold and a face-exclusion check, then cleaned up with Non-Maximum Suppression (NMS) before being drawn on screen.

## Requirements
- Python 3.9–3.11 (recommended)
- A working webcam
- Packages:
  - opencv-python
  - opencv-contrib-python (for `cv2.dnn.NMSBoxes`, if not already in opencv-python on your platform)
  - numpy
  - scikit-image
  - scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python numpy scikit-image scikit-learn
   ```

## Dataset
This project uses the **Deteksi Seragam** dataset from Roboflow. The dataset contains images of students with bounding box annotations for the uniform attributes.

**Dataset Link:** [Deteksi Seragam Dataset on Roboflow](https://universe.roboflow.com/skripsi-gztjx/deteksi-seragam/dataset/1)

### How to Download the Dataset in Pascal VOC XML Format
To run the training script successfully, you need the dataset annotated in **Pascal VOC XML** format. Follow these steps:

1. Click on the Dataset Link above. Create a free account or log in to Roboflow if you haven't already.
2. Navigate to the **Dataset** tab of the project.
3. Click on the **Download Dataset** button located in the top right area of the page.
4. In the "Format" dropdown menu, select **Pascal VOC**.
5. Select the **Download zip to computer** option (or obtain the download code if you prefer to download it directly via CLI/script).
6. Once the zip file is downloaded, extract it.
7. You should see `train`, `valid`, and `test` folders containing both the `.jpg` images and the `.xml` annotations.
8. Move the `train` and `test` folders directly into the root directory of this project.
   > Note: the current training/evaluation script (`main.py`) only reads from `train/` (for training) and `test/` (for evaluation). The `valid/` folder from the Roboflow export is not used by the script yet — you can leave it out, or keep it for future use (e.g. hyperparameter tuning).

## Project Structure
```
.
├── main.py                    # Main script: trains (or loads) the model, evaluates it, then runs live detection
├── svm_seragam_model.pkl      # Trained SVM model (auto-generated on first run, git-ignored)
├── train/                     # Training images + Pascal VOC XML annotations (from Roboflow, git-ignored)
├── test/                      # Test images + Pascal VOC XML annotations (from Roboflow, git-ignored)
├── valid/                     # Validation images + annotations from Roboflow (git-ignored, not yet used by main.py)
├── .gitignore
└── README.md
```

## Usage
Run the main script:
```bash
python main.py
```

What happens when you run it:
- **First run:** since `svm_seragam_model.pkl` doesn't exist yet, the script will train a new SVM model from the images in `train/`. This can take around **5–10 minutes** depending on your machine and dataset size. The trained model is then saved to `svm_seragam_model.pkl` so future runs skip training.
- **Every run:** the script evaluates the model against `test/` and prints a classification report (accuracy, precision/recall, mAP) to the terminal.
- **Then:** your webcam activates and live detection starts. A checklist overlay shows which uniform attributes (Dasi, Logo Sekolah, Ikat Pinggang, Topi) are currently detected.
- Press **`q`** to quit the application.

> Make sure to allow camera access when prompted by your OS.

## Notes & Limitations
- To retrain from scratch, delete `svm_seragam_model.pkl` and run the script again.
- Detection accuracy depends heavily on the dataset size/quality and lighting conditions during webcam use.
- The sliding-window + SVM approach is relatively lightweight but slower than modern deep-learning detectors (e.g. YOLO); frame processing is throttled (`process_every_n_frames`) to keep the webcam feed responsive.# Deteksi Atribut Seragam Sekolah (School Uniform Attribute Detection)

This project is a computer vision application that detects school uniform attributes (Dasi/Tie, Logo Sekolah/School Badge, Topi/Hat, and Ikat Pinggang/Belt) in real-time using a webcam feed. It utilizes Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier to identify the objects through a sliding window approach.

## Features
- Detects the following attributes:
  - Dasi (Tie)
  - Logo Sekolah (School Badge)
  - Topi (Hat)
  - Ikat Pinggang (Belt)
- Real-time webcam integration.
- Facial region exclusion to reduce false positives (using OpenCV Haar Cascades).
- Custom Non-Maximum Suppression (NMS) for refining overlapping bounding boxes.

## How It Works
1. **Feature extraction** — each object crop is resized to 96x96 and turned into a HOG (Histogram of Oriented Gradients) feature vector, which captures shape/edge patterns.
2. **Training** — an SVM classifier (linear kernel) is trained on these HOG features, using both the annotated objects and randomly sampled background patches as negative examples. Simple data augmentation (rotation, blur, noise) is applied to boost training data.
3. **Detection** — the webcam frame is scanned with a sliding window; each window is classified by the SVM, filtered by a confidence threshold and a face-exclusion check, then cleaned up with Non-Maximum Suppression (NMS) before being drawn on screen.

## Requirements
- Python 3.9–3.11 (recommended)
- A working webcam
- Packages:
  - opencv-python
  - opencv-contrib-python (for `cv2.dnn.NMSBoxes`, if not already in opencv-python on your platform)
  - numpy
  - scikit-image
  - scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python opencv-contrib-python numpy scikit-image scikit-learn
   ```

## Dataset
This project uses the **Deteksi Seragam** dataset from Roboflow. The dataset contains images of students with bounding box annotations for the uniform attributes.

**Dataset Link:** [Deteksi Seragam Dataset on Roboflow](https://universe.roboflow.com/skripsi-gztjx/deteksi-seragam/dataset/1)

### How to Download the Dataset in Pascal VOC XML Format
To run the training script successfully, you need the dataset annotated in **Pascal VOC XML** format. Follow these steps:

1. Click on the Dataset Link above. Create a free account or log in to Roboflow if you haven't already.
2. Navigate to the **Dataset** tab of the project.
3. Click on the **Download Dataset** button located in the top right area of the page.
4. In the "Format" dropdown menu, select **Pascal VOC**.
5. Select the **Download zip to computer** option (or obtain the download code if you prefer to download it directly via CLI/script).
6. Once the zip file is downloaded, extract it.
7. You should see `train`, `valid`, and `test` folders containing both the `.jpg` images and the `.xml` annotations.
8. Move the `train` and `test` folders directly into the root directory of this project.
   > Note: the current training/evaluation script (`main.py`) only reads from `train/` (for training) and `test/` (for evaluation). The `valid/` folder from the Roboflow export is not used by the script yet — you can leave it out, or keep it for future use (e.g. hyperparameter tuning).

## Project Structure
```
.
├── main.py                    # Main script: trains (or loads) the model, evaluates it, then runs live detection
├── svm_seragam_model.pkl      # Trained SVM model (auto-generated on first run, git-ignored)
├── train/                     # Training images + Pascal VOC XML annotations (from Roboflow, git-ignored)
├── test/                      # Test images + Pascal VOC XML annotations (from Roboflow, git-ignored)
├── valid/                     # Validation images + annotations from Roboflow (git-ignored, not yet used by main.py)
├── .gitignore
└── README.md
```

## Usage
Run the main script:
```bash
python main.py
```

What happens when you run it:
- **First run:** since `svm_seragam_model.pkl` doesn't exist yet, the script will train a new SVM model from the images in `train/`. This can take around **5–10 minutes** depending on your machine and dataset size. The trained model is then saved to `svm_seragam_model.pkl` so future runs skip training.
- **Every run:** the script evaluates the model against `test/` and prints a classification report (accuracy, precision/recall, mAP) to the terminal.
- **Then:** your webcam activates and live detection starts. A checklist overlay shows which uniform attributes (Dasi, Logo Sekolah, Ikat Pinggang, Topi) are currently detected.
- Press **`q`** to quit the application.

> Make sure to allow camera access when prompted by your OS.

## Notes & Limitations
- To retrain from scratch, delete `svm_seragam_model.pkl` and run the script again.
- Detection accuracy depends heavily on the dataset size/quality and lighting conditions during webcam use.
- The sliding-window + SVM approach is relatively lightweight but slower than modern deep-learning detectors (e.g. YOLO); frame processing is throttled (`process_every_n_frames`) to keep the webcam feed responsive.