# Vehicle Detection System

A Streamlit web application for detecting and classifying vehicles using YOLOv8.

## Features

- **Image Detection**: Upload and analyze images for vehicle detection
- **Video Detection**: Process videos and get annotated results
- **Real-time Statistics**: View detection counts and class distribution
- **Adjustable Parameters**: Configure confidence and IoU thresholds
- **Download Results**: Save annotated images and processed videos

## Vehicle Classes

The model detects 11 types of vehicles:
1. Auto Rickshaw
2. Cycle Rickshaw
3. CNG / Tempo
4. Bus
5. Jeep / SUV
6. Microbus
7. Minibus
8. Motorcycle
9. Truck
10. Private Sedan Car
11. Trailer

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your trained model (`yolov8s.pt`) is in the project directory

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

4. Use the interface to:
   - Upload images or videos
   - Adjust detection parameters
   - View results and statistics
   - Download processed files

## Configuration

### Detection Parameters

- **Confidence Threshold** (0.0 - 1.0): Minimum confidence score for detections (default: 0.25)
- **IoU Threshold** (0.0 - 1.0): IoU threshold for Non-Maximum Suppression (default: 0.45)

### Model Path

By default, the app looks for `yolov8s.pt` in the current directory. You can change this in the sidebar.

## Project Structure

```
vehicle/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── yolov8s.pt         # Trained YOLOv8 model
├── Inference -1.mp4   # Sample video 1
└── Inference -2.mp4   # Sample video 2
```

## Technical Details

- **Model**: YOLOv8s (Small variant)
- **Framework**: Ultralytics YOLOv8
- **Frontend**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Output Format**: Annotated images (JPG), videos (MP4)

## Tips for Best Results

1. Use high-quality images and videos
2. Adjust confidence threshold based on your needs (higher = fewer but more confident detections)
3. For crowded scenes, you may need to adjust the IoU threshold
4. Process shorter video clips for faster results
5. Ensure good lighting conditions in input media

## Troubleshooting

- If the model doesn't load, check the model path in the sidebar
- For slow video processing, try reducing the video resolution or length
- If detections are missing, try lowering the confidence threshold
- For duplicate detections, try increasing the IoU threshold

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- Webcam (optional, for future real-time detection features)

## License

This project uses YOLOv8 from Ultralytics, which is licensed under AGPL-3.0.
