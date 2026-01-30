"""
Batch Video Processing and Demo Generation
This script processes all sample videos and generates outputs for deliverables.

Author: arnab9961
Date: January 30, 2026
"""

import os
from pathlib import Path
from process_video import process_video_standalone
import time


def process_all_videos():
    """
    Process all sample videos in the project directory
    Creates outputs folder and processes Inference -1.mp4 and Inference -2.mp4
    """
    # Create outputs directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("VEHICLE DETECTION - BATCH VIDEO PROCESSING")
    print("="*70)
    print()
    
    # Model path
    model_path = "yolov10.pt"
    
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Please ensure yolov10.pt is in the project directory")
        return
    
    # List of videos to process
    videos = [
        "Inference -1.mp4",
        "Inference -2.mp4"
    ]
    
    all_stats = []
    total_start = time.time()
    
    for idx, video_file in enumerate(videos, 1):
        if not Path(video_file).exists():
            print(f"[WARNING] Video not found: {video_file}, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing Video {idx}/{len(videos)}: {video_file}")
        print(f"{'='*70}\n")
        
        # Generate output filename
        output_file = output_dir / f"processed_{video_file}"
        
        try:
            # Process video
            stats = process_video_standalone(
                input_path=video_file,
                output_path=str(output_file),
                model_path=model_path,
                conf=0.25,
                iou=0.45,
                save_stats=True
            )
            all_stats.append(stats)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {video_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos processed: {len(all_stats)}")
    print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Output directory: {output_dir.absolute()}")
    
    if all_stats:
        total_detections = sum(s['total_detections'] for s in all_stats)
        total_frames = sum(s['total_frames'] for s in all_stats)
        avg_fps = sum(s['average_fps'] for s in all_stats) / len(all_stats)
        
        print(f"\nAggregate Statistics:")
        print(f"  - Total frames processed: {total_frames:,}")
        print(f"  - Total vehicles detected: {total_detections:,}")
        print(f"  - Average processing FPS: {avg_fps:.2f}")
        
        # Combine class counts
        combined_classes = {}
        for stats in all_stats:
            for vehicle, count in stats['detections_by_class'].items():
                combined_classes[vehicle] = combined_classes.get(vehicle, 0) + count
        
        print(f"\nCombined Detection Breakdown:")
        for vehicle, count in sorted(combined_classes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections) * 100
            print(f"  - {vehicle}: {count} ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    print("All processing complete! ✓")
    print(f"{'='*70}\n")
    
    # Create README in outputs folder
    create_outputs_readme(output_dir, all_stats)


def create_outputs_readme(output_dir, stats_list):
    """
    Create a README file in the outputs directory explaining the contents
    """
    readme_content = """# Processed Videos - Vehicle Detection Output

## Contents

This directory contains the processed videos with vehicle detection annotations.

## Files

"""
    
    for stats in stats_list:
        input_name = Path(stats['input_video']).name
        output_name = Path(stats['output_video']).name
        
        readme_content += f"\n### {output_name}\n"
        readme_content += f"- **Source**: {input_name}\n"
        readme_content += f"- **Total Frames**: {stats['total_frames']:,}\n"
        readme_content += f"- **Processing Time**: {stats['processing_time_seconds']:.2f}s\n"
        readme_content += f"- **Average FPS**: {stats['average_fps']:.2f}\n"
        readme_content += f"- **Total Detections**: {stats['total_detections']:,}\n"
        readme_content += f"- **Confidence Threshold**: {stats['parameters']['confidence_threshold']}\n"
        readme_content += f"- **IoU Threshold**: {stats['parameters']['iou_threshold']}\n"
        
        readme_content += f"\n**Detection Breakdown**:\n"
        for vehicle, count in sorted(stats['detections_by_class'].items(), key=lambda x: x[1], reverse=True):
            readme_content += f"- {vehicle}: {count}\n"
        
        readme_content += f"\n**Statistics File**: `{output_name.replace('.mp4', '_stats.json')}`\n"
        readme_content += "\n---\n"
    
    readme_content += """
## How to View

1. Open the .mp4 files with any video player
2. Each frame shows bounding boxes around detected vehicles
3. Labels show vehicle type and confidence score
4. Check the corresponding _stats.json files for detailed statistics

## Model Information

- **Model**: YOLOv11
- **Classes**: 11 vehicle types (Auto Rickshaw, Cycle Rickshaw, CNG/Tempo, Bus, Jeep/SUV, Microbus, Minibus, Motorcycle, Truck, Private Sedan Car, Trailer)
- **Framework**: Ultralytics YOLO

## Notes

- Bounding box colors are randomly assigned per class
- Confidence scores shown as percentage (e.g., 0.85 = 85%)
- Multiple detections per frame are counted separately in statistics
- Statistics represent aggregate counts across all frames

Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"[INFO] Created README in outputs directory: {readme_path}")


if __name__ == "__main__":
    process_all_videos()
