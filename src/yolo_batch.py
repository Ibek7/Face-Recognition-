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

def process_images(image_files, model_name='yolov8n.pt', output_dir=None):
    """
    Process a batch of images using a YOLO model.
    """
    if not image_files:
        print('No images to process.')
        return

    model = YOLO(model_name)
    
    # The `save` argument will save results to a 'runs/detect/predict' directory
    # by default. We can't directly control the output path with the `save` flag here.
    # The output directory is managed by `ultralytics` library.
    # If you need to move results, you would do it after the prediction.
    detections = model(image_files, save=True)
    
    print(f"Processed {len(image_files)} images.")
    if output_dir:
         print(f"Results saved in the default 'runs/detect' directory. You can move them to '{output_dir}'.")
    else:
        print("Results saved in the default 'runs/detect' directory.")


def main():
    parser = argparse.ArgumentParser(description='YOLO Batch Image Processor')
    parser.add_argument('--input-dir', type=str, default='../data/images',
                        help='Directory containing images to process.')
    parser.add_argument('--output-dir', type=str, default='../runs/detect',
                        help='Directory to save detection results.')
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

    process_images(image_files, args.model, args.output_dir)

if __name__ == '__main__':
    main()
