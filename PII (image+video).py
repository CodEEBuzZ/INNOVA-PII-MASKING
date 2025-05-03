import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import re
import easyocr
import mediapipe as mp
from PIL import Image
from collections import Counter

# Initialize AI models
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
reader = easyocr.Reader(['en'])  # AI-based OCR

# Detection patterns
AADHAAR_PATTERN = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
PAN_PATTERN = r'[A-Z]{5}\d{4}[A-Z]{1}'
QR_DETECTOR = cv2.QRCodeDetector()

# Quality presets
QUALITY_PRESETS = {
    '360p': (640, 360),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4K': (3840, 2160)
}

def safe_blur(region, kernel_size=(35, 35), sigma=25):
    """Safe blur application with validation"""
    if region.size == 0:
        return region
    try:
        return cv2.GaussianBlur(region, kernel_size, sigma)
    except:
        return region

def detect_document_type(frame):
    """Document type detection with error handling"""
    try:
        if frame is None or frame.size == 0:
            return "Unknown"
        
        results = reader.readtext(frame, paragraph=False, batch_size=4)
        counts = Counter()
        
        for (_, text, prob) in results:
            if prob > 0.4:
                if re.search(AADHAAR_PATTERN, text):
                    counts['Aadhaar'] += 1
                if re.search(PAN_PATTERN, text):
                    counts['PAN'] += 1
        
        return counts.most_common(1)[0][0] if counts else "Unknown"
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return "Unknown"

def verify_document_type(input_path, is_video=True):
    """Document verification with frame validation"""
    try:
        if not is_video:
            frame = cv2.imread(input_path)
            if frame is None:
                return "Unknown", None
            return detect_document_type(frame), frame
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return "Unknown", None
            
        frames_to_check = min(5, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_skip = max(10, int(cap.get(cv2.CAP_PROP_FPS)))
        document_types = []
        sample_frame = None
        
        for i in range(frames_to_check):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            doc_type = detect_document_type(frame)
            if doc_type != "Unknown":
                document_types.append(doc_type)
                if sample_frame is None:
                    sample_frame = frame
        
        cap.release()
        
        if not document_types:
            return "Unknown", None
            
        return Counter(document_types).most_common(1)[0][0], sample_frame
    
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return "Unknown", None

def mask_frame(frame, document_type):
    """Safe frame masking with region validation"""
    try:
        if frame is None or frame.size == 0:
            return frame
            
        orig_h, orig_w = frame.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return frame
            
        work_frame = cv2.resize(frame, (640, int(640 * orig_h / orig_w)))
        
        # Face detection
        rgb_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)
        
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = work_frame.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                # Validate and adjust coordinates
                x, y = max(0, x-15), max(0, y-15)
                width = min(w-x, width+30)
                height = min(h-y, height+30)
                
                if width > 0 and height > 0:
                    work_frame[y:y+height, x:x+width] = safe_blur(work_frame[y:y+height, x:x+width], (45, 45), 30)
        
        # QR Code detection
        ret_qr, points = QR_DETECTOR.detect(work_frame)
        if ret_qr and points is not None:
            points = points.astype(int).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(points)
            work_frame[y:y+h, x:x+w] = safe_blur(work_frame[y:y+h, x:x+w], (35, 35), 25)
        
        # Document-specific masking
        text_results = reader.readtext(work_frame, paragraph=False, batch_size=8)
        for (bbox, text, prob) in text_results:
            if prob > 0.4:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                region_w = bottom_right[0] - top_left[0]
                region_h = bottom_right[1] - top_left[1]
                
                if region_w > 0 and region_h > 0:
                    if document_type == "Aadhaar" and re.search(AADHAAR_PATTERN, text):
                        mask_width = int(region_w * 0.66)
                        if mask_width > 0:
                            work_frame[top_left[1]:bottom_right[1], top_left[0]:top_left[0]+mask_width] = \
                                safe_blur(work_frame[top_left[1]:bottom_right[1], top_left[0]:top_left[0]+mask_width], (45, 45), 30)
                    elif document_type == "PAN" and re.search(PAN_PATTERN, text):
                        work_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                            safe_blur(work_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], (45, 45), 30)
        
        return cv2.resize(work_frame, (orig_w, orig_h)) if orig_w > 0 and orig_h > 0 else frame
    
    except Exception as e:
        print(f"Masking error: {str(e)}")
        return frame

def process_media(input_path, output_quality='720p', progress_callback=None):
    """Media processing with comprehensive error handling"""
    try:
        if not os.path.exists(input_path):
            return None, "Unknown"
            
        is_video = input_path.lower().endswith(('.mp4', '.mov', '.avi'))
        doc_type, _ = verify_document_type(input_path, is_video)
        
        if doc_type == "Unknown":
            return None, "Unknown"
        
        if not is_video:
            frame = cv2.imread(input_path)
            if frame is None:
                return None, "Unknown"
                
            masked_frame = mask_frame(frame, doc_type)
            out_w, out_h = QUALITY_PRESETS.get(output_quality, (frame.shape[1], frame.shape[0]))
            resized_frame = cv2.resize(masked_frame, (out_w, out_h))
            
            output_path = os.path.join(os.path.dirname(input_path), f"masked_{output_quality}.jpg")
            cv2.imwrite(output_path, resized_frame)
            return output_path, doc_type
        
        # Video processing
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return None, "Unknown"
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_w, out_h = QUALITY_PRESETS.get(output_quality, (orig_w, orig_h))
        aspect_ratio = out_w / out_h
        
        if (orig_w / orig_h) > aspect_ratio:
            out_h = int(out_w / (orig_w / orig_h))
        else:
            out_w = int(out_h * (orig_w / orig_h))
        
        output_path = os.path.join(os.path.dirname(input_path), f"masked_{output_quality}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            masked_frame = mask_frame(frame, doc_type)
            resized_frame = cv2.resize(masked_frame, (out_w, out_h))
            out.write(resized_frame)
            
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count / total_frames)
        
        cap.release()
        out.release()
        
        if frame_count == 0:
            return None, "Unknown"
            
        return output_path, doc_type
    
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None, "Unknown"

def main():
    # Page Configuration with improved aesthetics
    st.set_page_config(
        page_title="Document Masking Tool",
        layout="wide",
        page_icon="🔒",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            background-image: linear-gradient(to bottom, #D7BADE, #D3D3D3);
            color:#C71585;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stFileUploader {
            border: 2px dashed #4b5563;
            border-radius: 12px;
            padding: 2rem;
        }
        .stProgress>div>div>div>div {
            background-image: linear-gradient(to right, #4f46e5, #10b981);
        }
        .stAlert {
            border-radius: 8px;
        }
        .header-text {
            color: #1e40af;
        }
        .feature-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<h1 class="header-text">🔒 Smart Document Masking</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-card">
        <h3>Protect sensitive information with AI-powered masking</h3>
        <p>Automatically detects and redacts personal data from identity documents while preserving readability.</p>
        <div style="margin-top: 1rem;">
            <span style="display: inline-block; background: #e0e7ff; color: #4f46e5; padding: 0.25rem 0.75rem; border-radius: 9999px; margin-right: 0.5rem; font-size: 0.875rem;">Aadhaar Cards</span>
            <span style="display: inline-block; background: #d1fae5; color: #047857; padding: 0.25rem 0.75rem; border-radius: 9999px; margin-right: 0.5rem; font-size: 0.875rem;">PAN Cards</span>
            <span style="display: inline-block; background: #fce7f3; color: #9d174d; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Other IDs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    with st.container():
        st.markdown("### 📤 Upload Your Document")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse files",
            type=["mp4", "mov", "avi", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Supported formats: MP4, MOV, AVI, JPG, JPEG, PNG"
        )
    
    if uploaded_file:
        try:
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Original Document Preview
            with st.expander("👁️ Original Document Preview", expanded=True):
                if uploaded_file.type.startswith("video"):
                    st.video(input_path)
                else:
                    st.image(input_path, use_column_width=True)
            
            # Document Verification
            with st.status("🔍 Verifying document type...", expanded=True) as status:
                is_video = uploaded_file.type.startswith("video")
                doc_type, sample_frame = verify_document_type(input_path, is_video)
                
                if doc_type == "Unknown":
                    st.error("⚠️ Could not identify Aadhaar or PAN card. Please upload a clearer document.")
                    status.update(label="Verification failed", state="error")
                    return
                
                st.success(f"✅ Identified as: {doc_type} Card")
                status.update(label="Verification complete", state="complete")
                
                if sample_frame is not None:
                    with st.container():
                        st.image(sample_frame, caption="Sample Frame Analysis", use_container_width=True)
            
            # Confirmation Step
            st.markdown("### ✅ Confirmation")
            st.markdown("Is this the correct document you want to mask?")
            
            col1, col2, _ = st.columns([1,1,4])
            with col1:
                correct_btn = st.button("Yes, proceed with masking", 
                                      type="primary",
                                      use_container_width=True)
            with col2:
                incorrect_btn = st.button("No, upload different file",
                                         use_container_width=True)
            
            if incorrect_btn:
                st.warning("Please upload the correct document")
                st.rerun()
            
            if correct_btn:
                # Quality Selection
                st.markdown("### ⚙️ Output Settings")
                quality = st.radio(
                    "Select output quality:",
                    list(QUALITY_PRESETS.keys()),
                    horizontal=True,
                    index=1,
                    format_func=lambda x: f"{x} ({'▲' if x == 'High' else '►' if x == 'Medium' else '▼'} quality)"
                )
                
                # Processing Section
                with st.status(f"🛡️ Masking {doc_type} card...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    def progress_callback(p): 
                        progress_bar.progress(min(p, 1.0))
                    
                    output_path, detected_type = process_media(
                        input_path, 
                        output_quality=quality, 
                        progress_callback=progress_callback
                    )
                    
                    progress_bar.empty()
                    
                    if output_path is None:
                        st.error("❌ Processing failed. Please try again with a different file.")
                        status.update(label="Processing failed", state="error")
                        return
                    
                    status.update(label="Processing complete", state="complete")
                
                # Results Section
                st.success("🎉 Masking Complete!")
                
                
                with st.container():
                    st.markdown("### 🔍 Masked Document Preview")
                    if output_path.endswith(".mp4"):
                        st.video(output_path)
                    else:
                        img=Image.open(output_path)
                        st.image(img,output_path, output_format="PNG",width=img.width)
                
                # Download Section
                st.markdown("### 📥 Download Your Masked Document")
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Masked Output",
                        data=file,
                        file_name=f"masked_{os.path.basename(output_path)}",
                        mime="video/mp4" if output_path.endswith(".mp4") else "image/jpeg",
                        use_container_width=True,
                        help="Save your securely masked document"
                    )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(f"Main error: {str(e)}")

if __name__ == "__main__":
    main()