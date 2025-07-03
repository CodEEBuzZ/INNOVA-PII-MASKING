# PII Masking App

This project is a cutting-edge tool designed to detect and mask **Aadhar numbers**, **PAN numbers**, and **QR codes** in videos and photos while preserving non-sensitive visual elements. The app ensures robust privacy protection using AI-based techniques.

---

## **Purpose**
The primary objective of this project is to enhance data privacy by automatically identifying and masking sensitive personal information in multimedia content. It specifically focuses on:
- Detecting **Aadhar numbers**, **PAN numbers**, and **QR codes**.
- Blurring only sensitive information while keeping other visual elements intact.

---

## **Features**
- **Personal Information Masking**:
  - Detects and masks Aadhar numbers, PAN numbers, and QR codes.
  - Masks only the first 8 digits of an Aadhar number for privacy compliance.
  - Masks the whole PAN number for privacy compliance.
- **Selective Blurring**:
  - Sensitive information such as numbers, photos, and QR codes are blurred.
  - Faces, backgrounds, and non-sensitive content remain untouched.
- **Image Blurring**:
  - The app also supports PII detection and masking in still images.
- **AI-Powered Privacy**:
  - Utilizes advanced AI and image processing techniques for precision.

---

## **Technologies Used**
- **Python**: Core programming language.
- **OpenCV**: Video and image processing.
- **EasyOCR**: Optical Character Recognition for detecting text.
- **RegEx**: Pattern matching for number detection.
- **NumPy**: Data manipulation and computation.
- **Streamlit**: Web application framework.
- **Pillow**: Image processing.
- **MediaPipe**: Face detection and alignment.

---

## **Setup Instructions**
To get started with the PII Masking App:
1. Clone the repository:
   ```bash
   git clone https://github.com/CodEEBuzZ/INNOVA-PII-MASKING.git
   ```
2. Navigate to the project directory:
   ```bash
   cd INNOVA-PII-MASKING
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

---

## **Contributors**
This project was developed during **Innova 2025**, organized by **Entropy**, the coding club of GCETTS, hosted at **Geogo Tech Solutions**. The contributors are:
- **Anwesha Bhadury**
- **Atirath Pal**
- **Saikat Munshib**


---

## **Future Scope**
- Add support for detecting and masking **driving licenses** and other important documents.
- Enhance the system for **real-time PII detection and masking** during live video streams.
- Develop a **mobile application version** for quick and efficient on-the-go video PII masking from smartphones.

---
