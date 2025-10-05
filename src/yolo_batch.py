import argparse
import os
from ultralytics import YOLO

def get_image_files(data_dir):
    """
    Get a list of image files from a directory.
    Supported formats: .jpg, .jpeg, .png
    """
    image_files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(data_dir, f))
    return image_files


def process_images(image_files, model_name='yolov8n.pt', project_dir='runs/detect', name='predict'):
    """
    Process a batch of images using a YOLO model.
    project_dir/name controls the Ultralytics output directory.
    """
    if not image_files:
        print('No images to process.')
        return

    model = YOLO(model_name)

    # Use Ultralytics project/name to control output location
    # This will save results under {project_dir}/{name}
    model.predict(image_files, save=True, project=project_dir, name=name)

    print(f"Processed {len(image_files)} images. Results saved to '{project_dir}/{name}'.")



def main():
    parser = argparse.ArgumentParser(description='YOLO Batch Image Processor')
    parser.add_argument('--input-dir', type=str, default='data/images',
                        help='Directory containing images to process.')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Parent directory where results are saved.')
    parser.add_argument('--name', type=str, default='batch',
                        help='Sub-directory under project for this run.')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model to use for detection (e.g., yolov8n.pt).')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    image_files = get_image_files(args.input_dir)

    if not image_files:
        print(f'No images found in {args.input_dir}/')
        return

    process_images(image_files, args.model, args.project, args.name)


if __name__ == '__main__':
    main()
