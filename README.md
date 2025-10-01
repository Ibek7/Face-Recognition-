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

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Ibek7/Face-Recognition-.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Local Development

To run the script locally, you can use the following command:

```bash
python src/yolo_batch.py --input-dir data/images/
```

### Docker

To build and run the application using Docker, use the following commands:

1.  **Build the Docker image:**
    ```bash
    docker build -t face-recognition .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs face-recognition
    ```

### Docker Compose

For a more streamlined experience with Docker, you can use Docker Compose:

```bash
docker-compose up
```

This command will build the image (if it doesn't exist) and run the container, automatically mounting the necessary volumes.


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
