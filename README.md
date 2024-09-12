# Car License Plate Detection Using YOLOv10

This project implements and trains a YOLOv10 model to detect car license plates from images. YOLO (You Only Look Once) is a state-of-the-art object detection model known for its real-time processing capabilities. This project uses the YOLOv10n variant, which is lightweight and optimized for faster inference, to accurately detect license plates from vehicle images.

## Project Structure:
- **Dataset**: A custom dataset containing annotated images of car license plates.
- **Model**: YOLOv10n, chosen for its balance between speed and accuracy.
- **Training**: The model is trained for 100 epochs with a batch size of 16.
- **Evaluation**: Post-training evaluation using performance metrics like confusion matrices and F1 curves, alongside visualized predictions.

## Requirements:
- Python 3.x
- OpenCV
- PyTorch
- matplotlib
- YOLOv10 package (from GitHub repo)
- Google Colab (for training and inference)
- GPU support for efficient training

## Setup Instructions:
1. Clone the YOLOv10 repository:
    ```bash
    git clone https://github.com/THU-MIG/yolov10.git
    cd yolov10
    pip install .
    ```

2. Download YOLOv10 weight files:
    ```python
    import os
    import urllib.request

    # Create a directory for the weights
    weights_dir = os.path.join(os.getcwd(), "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Download the weight files
    urls = [
        "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",
        # Additional weights can be added here
    ]

    for url in urls:
        file_name = os.path.join(weights_dir, os.path.basename(url))
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded {file_name}")
    ```

3. Train the model with the car license plate dataset:
    ```bash
    yolo task=detect mode=train epochs=100 batch=16 plots=True model=weights/yolov10n.pt data=/content/custom_data.yaml
    ```

4. Evaluate the model by checking performance:
    - Confusion matrix
    - F1 curve
    - Visual results from validation batches

## Example Inference:
Once the model is trained, you can use it to make predictions on new car images to detect license plates:
```bash
yolo task=detect mode=predict conf=0.25 save=True model=weights/best.pt source=/path/to/test_image.jpg
```
## Results:
- Detailed evaluation includes:
  - Confusion matrix
  - F1 curves
  - Visual output of detected license plates in test images
