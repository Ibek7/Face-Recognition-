from ultralytics import YOLO
import os

# Path to the folder containing images
data_dir = 'images'

# Get list of image files (jpg, png, jpeg)
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f'No images found in {data_dir}/')
    exit(1)

# Load YOLOv8 nano model (fastest)
model = YOLO('yolov8n.pt')

# Run batch inference and save results
detections = model(image_files, save=True)

print(f"Processed {len(image_files)} images. Results saved in 'runs/detect/predict'.")
