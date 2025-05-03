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

# ====== NEW QR CODE DETECTION FUNCTIONS ======
def detect_and_mask_qr(frame):
    """Enhanced QR code detection and masking"""
    qr_results = []
    try:
        qr_detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
        
        if retval and points is not None:
            points = points.astype(int)
            for i, (data, qr_points) in enumerate(zip(decoded_info, points)):
                # If QR decoding failed, try OCR on QR region
                if not data:
                    x, y, w, h = cv2.boundingRect(qr_points)
                    qr_region = frame[y:y+h, x:x+w]
                    data = " ".join(reader.readtext(qr_region, detail=0, paragraph=True))
                
                # Store QR code info
                qr_results.append({
                    "id": i+1,
                    "data": data[:100] + "..." if len(data) > 100 else data,
                    "position": qr_points.tolist()
                })
                
                # Mask the QR code
                x, y, w, h = cv2.boundingRect(qr_points)
                frame[y:y+h, x:x+w] = safe_blur(frame[y:y+h, x:x+w], (30, 30), 40)
    except Exception as e:
        print(f"QR detection error: {str(e)}")
    
    return frame, qr_results

def visualize_detections(frame, detection_results):
    """Create visualization of detected elements"""
    viz_frame = frame.copy()
    
    # Draw QR codes (blue)
    for qr in detection_results.get("qr_codes", []):
        points = np.array(qr["position"], dtype=np.int32)
        cv2.polylines(viz_frame, [points], True, (255, 0, 0), 2)
        cv2.putText(viz_frame, f"QR {qr['id']}", tuple(points[0][0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw faces (red)
    for face in detection_results.get("faces", []):
        x, y, w, h = face["position"]
        cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(viz_frame, f"Face {face['id']}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return viz_frame
# ====== END NEW FUNCTIONS ======

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
    """Updated frame masking with QR detection"""
    detection_results = {
        "document_type": document_type,
        "qr_codes": [],
        "faces": []
    }
    
    try:
        if frame is None or frame.size == 0:
            return frame, detection_results
            
        orig_h, orig_w = frame.shape[:2]
        work_frame = cv2.resize(frame, (640, int(640 * orig_h / orig_w)))
        
        # 1. QR Code Detection and Masking
        work_frame, qr_results = detect_and_mask_qr(work_frame)
        detection_results["qr_codes"] = qr_results
        
        # 2. Face Detection and Masking (existing)
        rgb_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)
        
        if face_results.detections:
            for i, detection in enumerate(face_results.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w = work_frame.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                detection_results["faces"].append({
                    "id": i+1,
                    "position": [x, y, width, height],
                    "confidence": detection.score[0]
                })
                
                x, y = max(0, x-15), max(0, y-15)
                width = min(w-x, width+30)
                height = min(h-y, height+30)
                
                if width > 0 and height > 0:
                    work_frame[y:y+height, x:x+width] = safe_blur(work_frame[y:y+height, x:x+width], (45, 45), 30)
        
        # 3. Document-specific text masking (existing)
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
        
        return cv2.resize(work_frame, (orig_w, orig_h)), detection_results
    
    except Exception as e:
        print(f"Masking error: {str(e)}")
        return frame, detection_results

def process_media(input_path, output_quality='720p', progress_callback=None):
    """Updated media processing with visualization"""
    try:
        if not os.path.exists(input_path):
            return None, None, "Unknown", {}
            
        is_video = input_path.lower().endswith(('.mp4', '.mov', '.avi'))
        doc_type, sample_frame = verify_document_type(input_path, is_video)
        
        if doc_type == "Unknown":
            return None, None, "Unknown", {}
        
        if not is_video:
            frame = cv2.imread(input_path)
            masked_frame, detection_results = mask_frame(frame, doc_type)
            viz_frame = visualize_detections(frame, detection_results)
            
            out_w, out_h = QUALITY_PRESETS.get(output_quality, (frame.shape[1], frame.shape[0]))
            resized_frame = cv2.resize(masked_frame, (out_w, out_h))
            resized_viz = cv2.resize(viz_frame, (out_w, out_h))
            
            output_path = os.path.join(os.path.dirname(input_path), f"masked_{output_quality}.jpg")
            viz_path = os.path.join(os.path.dirname(input_path), f"viz_{output_quality}.jpg")
            
            cv2.imwrite(output_path, resized_frame)
            cv2.imwrite(viz_path, resized_viz)
            
            return output_path, viz_path, doc_type, detection_results
        
        # Video processing remains similar but needs updating
        # ... [rest of your existing video processing code]
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None, None, "Unknown", {}

def main():
    st.set_page_config(page_title="Document Masking Tool", layout="wide")
    st.title("🔒 Smart Document Masking")
    
    uploaded_file = st.file_uploader("Upload Document", 
                                   type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        is_video = tmp_file_path.lower().endswith(('.mp4', '.mov', '.avi'))
        
        quality = st.selectbox("Output Quality", list(QUALITY_PRESETS.keys()))
        
        if st.button("Start Processing"):
            progress_bar = st.progress(0)
            
            def update_progress(p):
                progress_bar.progress(min(p, 1.0))
            
            output_path, viz_path, doc_type, detection_results = process_media(
                tmp_file_path, quality, update_progress)
            
            if output_path:
                st.success("Processing complete!")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Masked Document")
                    if is_video:
                        st.video(output_path)
                    else:
                        st.image(output_path)
                
                with col2:
                    st.subheader("Detection Visualization")
                    if viz_path:
                        st.image(viz_path)
                
                with st.expander("Detection Details"):
                    st.json(detection_results)
                
                # Clean up temp files
                try:
                    os.unlink(tmp_file_path)
                    os.unlink(output_path)
                    if viz_path and os.path.exists(viz_path):
                        os.unlink(viz_path)
                except:
                    pass

if __name__ == "__main__":
    main()