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
from ultralytics import YOLO

# Load a pretrained YOLO model (e.g., YOLOv8)
model = YOLO('yolov8n.pt')  # You can use 'yolov8n.pt', 'yolov8s.pt', etc. based on your requirements.
if os.path.exists(r"C:\Users\Saikat Munshib\runs\obb\train2\weights\best.pt"):
    model = YOLO(r"C:\Users\Saikat Munshib\runs\obb\train2\weights\best.pt")
else:
    model_path = r"C:\Users\Saikat Munshib\runs\obb\train2\weights\best.pt"
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Initialize AI models
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
reader = easyocr.Reader(['en'])  # AI-based OCR

# Enhanced detection patterns
AADHAAR_PATTERN = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
PAN_PATTERN = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'  # More strict PAN pattern
PAN_KEYWORDS = ['income', 'tax', 'department', 'permanent', 'account', 'number']

# Quality presets
QUALITY_PRESETS = {
    '360p': (640, 360),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4K': (3840, 2160)
}

def detect_pan_card(frame):
    """Specialized PAN card detection with multiple checks"""
    try:
        # First check for PAN number pattern
        results = reader.readtext(frame, paragraph=False, batch_size=8)
        pan_numbers = [text for (_, text, prob) in results 
                      if prob > 0.4 and re.search(PAN_PATTERN, text)]
        
        if pan_numbers:
            return True
        
        # Then check for PAN-related keywords
        text = " ".join([text for (_, text, _) in results])
        if any(keyword in text.lower() for keyword in PAN_KEYWORDS):
            return True
            
        # Finally check visual features (PAN card has specific layout)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # PAN cards usually have rectangular contours with specific aspect ratio
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 1.5 < aspect_ratio < 2.5 and w > 100 and h > 50:  # Typical PAN card dimensions
                return True
                
        return False
    except:
        return False

def detect_document_type(frame):
    """Enhanced document type detection with PAN-specific checks"""
    try:
        if frame is None or frame.size == 0:
            return "Unknown"
        
        # YOLO object detection
        results = model(frame)
        boxes = results.xywh[0][:, :-1].cpu().numpy()  # Get bounding boxes

        # Print or log the detected boxes to see if YOLO detects the right areas
        print("Detected boxes:", boxes)

        # First check for PAN card (more specific checks)
        if detect_pan_card(frame):
            return "PAN"
            
        # Then check for Aadhaar QR code
        qr_points = cv2.QRCodeDetector().detect(frame)[0]
        if qr_points is not None:
            x, y, w, h = cv2.boundingRect(qr_points.astype(int))
            qr_region = frame[y:y+h, x:x+w]
            qr_text = reader.readtext(qr_region, detail=0, paragraph=True)
            if qr_text and any("uidai" in t.lower() for t in qr_text):
                return "Aadhaar"
        
        # Fallback to OCR for Aadhaar numbers
        results = reader.readtext(frame, paragraph=False, batch_size=8)
        aadhaar_count = sum(1 for (_, text, prob) in results 
                           if prob > 0.4 and re.search(AADHAAR_PATTERN, text))
        
        return "Aadhaar" if aadhaar_count > 0 else "Unknown"
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return "Unknown"


def verify_document_type(input_path, is_video=True):
    """Verification with PAN-specific checks"""
    try:
        if not is_video:
            frame = cv2.imread(input_path)
            if frame is None:
                st.error("Failed to load image.")
                return "Unknown", None
            return detect_document_type(frame), frame
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Failed to open video.")
            return "Unknown", None
            
        # Check first, middle, last frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check_frames = [0, total_frames//2, total_frames-1]
        
        document_types = []
        sample_frame = None
        
        for frame_pos in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
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
            st.error("No document detected in the video.")
            return "Unknown", None
            
        return Counter(document_types).most_common(1)[0][0], sample_frame
    
    except Exception as e:
        st.error(f"Verification error: {str(e)}")
        return "Unknown", None

def mask_frame(frame, document_type):
    """Document-specific masking with PAN focus"""
    try:
        if frame is None or frame.size == 0:
            return frame
        
        # Work at lower resolution for speed
        h, w = frame.shape[:2]
        work_frame = cv2.resize(frame, (640, int(640 * h / w)))
        
        # Mask faces
        rgb_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)
        
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y = int(bbox.xmin * work_frame.shape[1]), int(bbox.ymin * work_frame.shape[0])
                width, height = int(bbox.width * work_frame.shape[1]), int(bbox.height * work_frame.shape[0])
                x, y = max(0, x-15), max(0, y-15)
                width, height = min(work_frame.shape[1]-x, width+30), min(work_frame.shape[0]-y, height+30)
                if width > 0 and height > 0:
                    work_frame[y:y+height, x:x+width] = cv2.GaussianBlur(
                        work_frame[y:y+height, x:x+width], (45, 45), 30)
        
        # Mask text regions
        if document_type == "PAN":
            # Print the detected text regions
            results = reader.readtext(work_frame, paragraph=False, batch_size=8)
            for (bbox, text, prob) in results:
                print(f"Detected text: {text} with probability {prob}")
                if prob > 0.3:  # Lower confidence threshold for PAN
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    work_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                        cv2.GaussianBlur(
                            work_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], 
                            (55, 55), 40)
        
        elif document_type == "Aadhaar":
            results = reader.readtext(work_frame, paragraph=False, batch_size=8)
            for (bbox, text, prob) in results:
                print(f"Detected text: {text} with probability {prob}")
                if prob > 0.4:
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    work_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                        cv2.GaussianBlur(
                            work_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], 
                            (45, 45), 30)
        
        return cv2.resize(work_frame, (w, h))
    
    except Exception as e:
        print(f"Masking error: {str(e)}")
        return frame


def process_image(input_path, quality, update_progress):
    try:
        frame = cv2.imread(input_path)
        if frame is None:
            st.error("Failed to load image.")
            return None, None

        doc_type = detect_document_type(frame)
        masked_frame = mask_frame(frame, doc_type)
        height, width = QUALITY_PRESETS[quality]
        resized = cv2.resize(masked_frame, (width, height))

        output_path = input_path.replace(".", "_masked.")
        cv2.imwrite(output_path, resized)
        update_progress(1.0)
        return output_path, doc_type

    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None, None

def process_video(input_path, quality, update_progress):
    try:
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = QUALITY_PRESETS[quality]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_masked{ext}"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            document_type = detect_document_type(frame)
            masked_frame = mask_frame(frame, document_type)
            resized_frame = cv2.resize(masked_frame, (width, height))

            out.write(resized_frame)

            frame_idx += 1
            update_progress(min(1.0, frame_idx / total_frames))

        cap.release()
        out.release()
        return output_path, document_type

    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        return None, None


def main():
    st.set_page_config(page_title="Document Masking Tool", layout="wide")
    st.title("🔒 Smart Document Masking")
    st.markdown("""**Automatically detects and masks sensitive information in:**
    - PAN Cards (improved detection)
    - Aadhaar Cards
    """)

    uploaded_file = st.file_uploader("Upload Document", 
                                   type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        is_video = tmp_file_path.lower().endswith(('.mp4', '.mov', '.avi'))
        
        st.subheader("Processing Options")
        col1, col2 = st.columns(2)
        with col1:
            quality = st.selectbox("Select Output Quality", options=['360p', '720p', '1080p', '4K'])
        with col2:
            process_btn = st.button("Start Processing")
        
        if process_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing... {int(progress*100)}%")
            
            # Process the file (image or video)
            if is_video:
                status_text.text("Processing video...")
                result_path, doc_type = process_video(tmp_file_path, quality, update_progress)
            else:
                status_text.text("Processing image...")
                result_path, doc_type = process_image(tmp_file_path, quality, update_progress)

            if result_path:
                status_text.text("Processing complete!")
                st.subheader(f"Document Type: {doc_type}")
                if is_video:
                    st.video(result_path)
                else:
                    st.image(result_path)
                
                with open(result_path, "rb") as file:
                    st.download_button(
                        label="Download Processed File",
                        data=file,
                        file_name=os.path.basename(result_path),
                        mime="video/mp4" if is_video else "image/jpeg"
                    )
                
                # Clean up temporary files
                try:
                    os.unlink(tmp_file_path)
                    os.unlink(result_path)
                except:
                    pass

if __name__ == "__main__":
    main()