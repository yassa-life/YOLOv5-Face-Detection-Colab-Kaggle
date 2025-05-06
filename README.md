# Face Detection with YOLOv5 and Kaggle

## Overview
This repository contains a YOLOv5-based face detection model trained on the [Face Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset) from Kaggle. The project includes a Jupyter notebook (`trainedmodel.ipynb`) that demonstrates the process of setting up the environment, downloading the dataset, training the YOLOv5 model, and performing inference on a test image.

## Features
- **YOLOv5 Implementation**: Utilizes the YOLOv5 model for efficient and accurate face detection.
- **Kaggle Dataset Integration**: Downloads and processes the Face Detection Dataset using Kaggle API.
- **Training Pipeline**: Includes code for training the model with 20 epochs, achieving a mAP50 of 0.894 and mAP50-95 of 0.588.
- **Inference Example**: Demonstrates face detection on a sample image (`test.jpg`) using the trained model.
- **Google Colab Compatibility**: The notebook is designed to run on Google Colab with GPU support, with optional Google Drive integration.

## Prerequisites
- Python 3.11+
- PyTorch 2.6.0+ with CUDA support
- Google Colab or a local environment with GPU (optional for faster training)
- Kaggle account and API token (`kaggle.json`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/YOLOv5-Face-Detection.git
   cd YOLOv5-Face-Detection
   ```

2. **Set Up Kaggle API**:
   - Download your `kaggle.json` file from Kaggle.
   - Upload it to the root directory of the project or follow the notebook instructions to upload it in Colab.

3. **Install Dependencies**:
   Run the following commands in the notebook or terminal:
   ```bash
   pip install -q kaggle
   pip install -r requirements.txt  # If requirements.txt is provided
   ```

4. **Download YOLOv5**:
   The notebook clones the YOLOv5 repository. Ensure you have it in your working directory:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   ```

## Usage
1. **Open the Notebook**:
   - Upload `trainedmodel.ipynb` to Google Colab or run it locally with Jupyter.
   - Follow the cells to:
     - Upload `kaggle.json` for dataset access.
     - Download and unzip the Face Detection Dataset.
     - Train the YOLOv5 model.
     - Perform inference on a test image.

2. **Storage Options**:
   - **Google Drive (Optional)**: Mount Google Drive in Colab to save datasets, model weights, and results for persistence across sessions. Update paths in the notebook to use `/content/drive/MyDrive/`.
   - **Colab Temporary Storage**: Use Colab's temporary storage (`/content/`) to save files without Google Drive. Note that files in `/content/` are deleted after the session ends.

3. **Training**:
   - The notebook trains the model for 20 epochs with the following command (update paths based on storage choice):
     ```bash
     !python /content/yolov5/train.py --img 640 --batch 16 --epochs 20 --data /content/yolov5/data/custom_data.yaml --weights yolov5s.pt --cache
     ```
   - Results are saved in `runs/train/exp2` (or update to `/content/drive/MyDrive/yolov5/runs/train/exp2` if using Drive).

4. **Inference**:
   - Run inference on a test image (`test.jpg`) using (update paths accordingly):
     ```bash
     !python /content/yolov5/detect.py --weights /content/yolov5/runs/train/exp2/weights/best.pt --source /content/test.jpg
     ```
   - Results are saved in `runs/detect/exp4` (or update to `/content/drive/MyDrive/yolov5/runs/detect/exp4` if using Drive).

## Dataset
- **Source**: [Face Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset)
- **License**: CC0-1.0
- **Description**: Contains images with labeled bounding boxes for faces, suitable for object detection tasks.
- **Usage**: The notebook downloads the dataset using the Kaggle API and unzips it to the `dataset` directory (e.g., `/content/dataset` or `/content/drive/MyDrive/dataset`).

## Model Performance
- **Training Results** (after 20 epochs):
  - Precision: 0.893
  - Recall: 0.824
  - mAP50: 0.894
  - mAP50-95: 0.588
- **Model Weights**: Saved as `best.pt` and `last.pt` in `runs/train/exp2/weights`.

## Directory Structure
```
YOLOv5-Face-Detection/
│
├── trainedmodel.ipynb        # Main notebook for training and inference
├── kaggle.json               # Kaggle API token (not included, user must provide)
├── test.jpg                  # Sample test image for inference
├── yolov5/                   # YOLOv5 repository (cloned during execution)
├── dataset/                  # Unzipped Face Detection Dataset
├── runs/                     # Training and detection results
│   ├── train/exp2/           # Training logs and weights
│   └── detect/exp4/          # Detection results
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the model implementation.
- [Kaggle Face Detection Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset) for providing the training data.
- Google Colab for providing free GPU resources.
