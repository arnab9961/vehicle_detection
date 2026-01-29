import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import time

# Page configuration
st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stats-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Vehicle classes
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

@st.cache_resource
def load_model(model_path):
    """Load the YOLOv8 model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, conf_threshold, iou_threshold):
    """Process a single image and return results"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # Get the annotated image
    annotated_img = results[0].plot()
    
    # Convert BGR to RGB
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Extract detection statistics
    boxes = results[0].boxes
    class_counts = {}
    total_detections = len(boxes)
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    return annotated_img, class_counts, total_detections

def process_video(video_path, model, conf_threshold, iou_threshold, progress_bar, status_text):
    """Process video and return path to processed video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_detections = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
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
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    return output_path, all_detections

def main():
    # Title and description
    st.title("🚗 Vehicle Detection System")
    st.markdown("### Real-time vehicle detection using YOLOv8")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_path = st.text_input(
            "Model Path",
            value="yolov8s.pt",
            help="Path to your trained YOLOv8 model"
        )
        
        # Detection parameters
        st.subheader("Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        iou_threshold = st.slider(
            "IOU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="IoU threshold for NMS"
        )
        
        st.markdown("---")
        
        # Class information
        st.subheader("📋 Vehicle Classes")
        with st.expander("View all classes"):
            for idx, class_name in CLASSES.items():
                st.text(f"{idx}: {class_name}")
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please make sure the model path is correct.")
        return
    
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "ℹ️ About"])
    
    # Image Detection Tab
    with tab1:
        st.header("Image Vehicle Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                help="Upload an image for vehicle detection"
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Image", use_container_width=True)
                
                if st.button("🔍 Detect Vehicles", key="detect_image"):
                    with st.spinner("Processing image..."):
                        annotated_img, class_counts, total_detections = process_image(
                            image, model, conf_threshold, iou_threshold
                        )
                        
                        # Store results in session state
                        st.session_state['annotated_img'] = annotated_img
                        st.session_state['class_counts'] = class_counts
                        st.session_state['total_detections'] = total_detections
        
        with col2:
            if 'annotated_img' in st.session_state:
                st.subheader("Detection Results")
                st.image(
                    st.session_state['annotated_img'],
                    caption="Detected Vehicles",
                    use_container_width=True
                )
                
                # Download button
                result_img = Image.fromarray(st.session_state['annotated_img'])
                buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                result_img.save(buf.name)
                
                with open(buf.name, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Result",
                        data=f,
                        file_name="detected_vehicles.jpg",
                        mime="image/jpeg"
                    )
        
        # Statistics
        if 'class_counts' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Detection Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Detections", st.session_state['total_detections'])
            
            with col2:
                st.metric("Unique Classes", len(st.session_state['class_counts']))
            
            with col3:
                if st.session_state['class_counts']:
                    most_common = max(st.session_state['class_counts'].items(), key=lambda x: x[1])
                    st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")
            
            # Detailed breakdown
            if st.session_state['class_counts']:
                st.subheader("Class Distribution")
                for class_name, count in sorted(st.session_state['class_counts'].items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{class_name}:** {count}")
    
    # Video Detection Tab
    with tab2:
        st.header("Video Vehicle Detection")
        
        # Option to choose between upload or existing videos
        video_source = st.radio(
            "Select video source:",
            ["Upload Video", "Use Existing Video"],
            horizontal=True
        )
        
        video_path = None
        
        if video_source == "Upload Video":
            uploaded_video = st.file_uploader(
                "Choose a video...",
                type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
                help="Upload a video for vehicle detection"
            )
            
            if uploaded_video is not None:
                # Save uploaded video temporarily
                file_extension = uploaded_video.name.split('.')[-1]
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
                tfile.write(uploaded_video.read())
                video_path = tfile.name
        else:
            # List existing videos in the directory
            existing_videos = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if existing_videos:
                selected_video = st.selectbox(
                    "Select a video:",
                    existing_videos
                )
                video_path = selected_video
            else:
                st.warning("No existing videos found in the directory.")
        
        if video_path is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Video")
                st.video(video_path)
            
            if st.button("🔍 Detect Vehicles in Video", key="detect_video"):
                with col2:
                    st.subheader("Processing...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    output_path, all_detections = process_video(
                        video_path, model, conf_threshold, iou_threshold,
                        progress_bar, status_text
                    )
                    processing_time = time.time() - start_time
                    
                    status_text.text(f"✅ Processing complete! Time: {processing_time:.2f}s")
                    
                    # Store in session state
                    st.session_state['video_output'] = output_path
                    st.session_state['video_detections'] = all_detections
                    st.session_state['video_processing_time'] = processing_time
            
            # Show results
            if 'video_output' in st.session_state:
                with col2:
                    st.subheader("Detected Video")
                    st.video(st.session_state['video_output'])
                    
                    # Download button
                    with open(st.session_state['video_output'], 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=f,
                            file_name="detected_vehicles.mp4",
                            mime="video/mp4"
                        )
                
                # Video statistics
                st.markdown("---")
                st.subheader("📊 Video Detection Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_video_detections = sum(st.session_state['video_detections'].values())
                    st.metric("Total Detections", total_video_detections)
                
                with col2:
                    st.metric("Unique Classes", len(st.session_state['video_detections']))
                
                with col3:
                    st.metric("Processing Time", f"{st.session_state['video_processing_time']:.2f}s")
                
                # Detailed breakdown
                if st.session_state['video_detections']:
                    st.subheader("Class Distribution (All Frames)")
                    for class_name, count in sorted(st.session_state['video_detections'].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{class_name}:** {count}")
    
    # About Tab
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### 🚗 Vehicle Detection System
        
        This application uses a trained YOLOv8 model to detect and classify vehicles in images and videos.
        
        #### 📋 Supported Vehicle Classes:
        - **Auto Rickshaw** - Three-wheeled motorized vehicles
        - **Cycle Rickshaw** - Three-wheeled pedal-powered vehicles
        - **CNG / Tempo** - Compressed Natural Gas vehicles
        - **Bus** - Large passenger vehicles
        - **Jeep / SUV** - Sport Utility Vehicles
        - **Microbus** - Small buses
        - **Minibus** - Medium-sized buses
        - **Motorcycle** - Two-wheeled motorized vehicles
        - **Truck** - Large cargo vehicles
        - **Private Sedan Car** - Passenger cars
        - **Trailer** - Cargo trailers
        
        #### 🔧 Features:
        - **Real-time Detection**: Process images and videos with YOLOv8
        - **Adjustable Parameters**: Configure confidence and IoU thresholds
        - **Statistics**: View detailed detection statistics
        - **Download Results**: Save annotated images and videos
        
        #### 📊 Model Information:
        - **Model**: YOLOv8s (Small variant)
        - **Classes**: 11 vehicle types
        - **Framework**: Ultralytics YOLOv8
        
        #### 💡 Usage Tips:
        1. Adjust the **confidence threshold** to filter out low-confidence detections
        2. Modify the **IoU threshold** to control overlapping detections
        3. Use high-quality images/videos for better results
        4. Process shorter video clips for faster results
        
        ---
        **Built with ❤️ using Streamlit and YOLOv8**
        """)

if __name__ == "__main__":
    main()
