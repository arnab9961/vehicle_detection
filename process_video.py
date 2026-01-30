"""
Vehicle Detection System - Video Processing Script
This script processes videos with vehicle detection and creates annotated output.
Can be used independently or integrated with the Streamlit application.

Author: arnab9961
Date: January 30, 2026
"""

import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path
import time
import json

# Vehicle classes mapping
CLASSES = {
    0: "Auto Rickshaw",
    1: "Cycle Rickshaw",
    2: "CNG / Tempo",
    3: "Bus",
    4: "Jeep / SUV",
    5: "Microbus",
    6: "Minibus",
    7: "Motorcycle",
    8: "Truck",
    9: "Private Sedan Car",
    10: "Trailer"
}


def process_video_standalone(input_path, output_path, model_path, conf=0.25, iou=0.45, save_stats=False):
    """
    Process a video file with vehicle detection
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save output video
        model_path (str): Path to YOLO model weights
        conf (float): Confidence threshold (default: 0.25)
        iou (float): IoU threshold for NMS (default: 0.45)
        save_stats (bool): Whether to save statistics to JSON file
        
    Returns:
        dict: Statistics including detection counts and processing time
    """
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"[INFO] Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {total_frames/fps:.2f} seconds")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracking variables
    frame_count = 0
    all_detections = {}
    start_time = time.time()
    
    print(f"[INFO] Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Count detections
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")
            all_detections[cls_name] = all_detections.get(cls_name, 0) + 1
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames}) - ETA: {eta:.1f}s")
    
    # Clean up
    cap.release()
    out.release()
    
    # Calculate statistics
    processing_time = time.time() - start_time
    avg_fps = frame_count / processing_time
    total_detections = sum(all_detections.values())
    
    stats = {
        'input_video': input_path,
        'output_video': output_path,
        'total_frames': frame_count,
        'processing_time_seconds': round(processing_time, 2),
        'average_fps': round(avg_fps, 2),
        'total_detections': total_detections,
        'detections_by_class': all_detections,
        'parameters': {
            'confidence_threshold': conf,
            'iou_threshold': iou,
            'model': model_path
        }
    }
    
    print(f"\n[SUCCESS] Video processing complete!")
    print(f"  - Output saved to: {output_path}")
    print(f"  - Processing time: {processing_time:.2f}s")
    print(f"  - Average FPS: {avg_fps:.2f}")
    print(f"  - Total detections: {total_detections}")
    print(f"\nDetection breakdown:")
    for vehicle, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {vehicle}: {count}")
    
    # Save statistics to JSON if requested
    if save_stats:
        stats_path = output_path.replace('.mp4', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n[INFO] Statistics saved to: {stats_path}")
    
    return stats


def main():
    """
    Main function to handle command-line arguments and execute video processing
    """
    parser = argparse.ArgumentParser(
        description='Vehicle Detection Video Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_video.py -i input.mp4 -o output.mp4
  
  # With custom parameters
  python process_video.py -i input.mp4 -o output.mp4 -c 0.35 -iou 0.5
  
  # Save statistics
  python process_video.py -i input.mp4 -o output.mp4 --save-stats
  
  # Use custom model
  python process_video.py -i input.mp4 -o output.mp4 -m yolov11l.pt
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Path to input video file')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to save output video file')
    parser.add_argument('-m', '--model', default='yolov10.pt',
                        help='Path to YOLO model weights (default: yolov10.pt)')
    parser.add_argument('-c', '--conf', type=float, default=0.25,
                        help='Confidence threshold 0.0-1.0 (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS 0.0-1.0 (default: 0.45)')
    parser.add_argument('--save-stats', action='store_true',
                        help='Save detection statistics to JSON file')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.input).exists():
        print(f"[ERROR] Input video not found: {args.input}")
        return
    
    if not Path(args.model).exists():
        print(f"[ERROR] Model file not found: {args.model}")
        return
    
    # Validate parameters
    if not 0.0 <= args.conf <= 1.0:
        print(f"[ERROR] Confidence threshold must be between 0.0 and 1.0")
        return
    
    if not 0.0 <= args.iou <= 1.0:
        print(f"[ERROR] IoU threshold must be between 0.0 and 1.0")
        return
    
    # Process video
    try:
        process_video_standalone(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            save_stats=args.save_stats
        )
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
