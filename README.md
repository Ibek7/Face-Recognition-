# YOLO Batch Inference

# Face Recognition Project

This project is a face recognition application that uses the YOLO (You Only Look Once) object detection model.

## Project Structure

- `data/`: Contains directories for images and videos.
- `models/`: Stores trained model files.
- `notebooks/`: Jupyter notebooks for experimentation and analysis.
- `src/`: Source code for the application.
  - `utils/`: Utility functions.
- `tests/`: Unit tests for the project.
- `yolo_batch.py`: Main script for batch processing.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Specifies intentionally untracked files to ignore.

## Quickstart

### Local (venv)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python src/yolo_batch.py --input-dir data/images --project runs/detect --name batch
```

### Docker

```bash
docker build -t face-recognition:latest .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/runs:/app/runs" face-recognition:latest
```

### Docker Compose

```bash
docker compose up --build
```

## Notes

- Place images under `data/images`.
- Results will be saved to `runs/detect/batch` by default.

## Requirements

- Python 3.8 or higher
- Ultralytics YOLOv8

## Installation

```bash
pip install ultralytics
```

## Usage

1. Place your images in a folder named `images/` in the project directory.
2. Run the script:

```bash
python yolo_batch.py
```

3. Detection results will be saved in the `runs/detect/predict` directory.

## Notes

- The script uses the default YOLOv8n (nano) model for speed and simplicity.
- You can change the model or parameters in the script as needed.
