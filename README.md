# 🧠 Face Recognition with YuNet in Google Colab

This project implements a face recognition and attendance logging system using OpenCV's YuNet model within a Google Colab environment. It captures images through the user's webcam, detects faces in real time, and logs essential details such as roll number, date, time, number of faces detected, and the captured image.

## 🚀 Features

- 📷 Captures photos directly via webcam using JavaScript in Colab
- 🧠 Uses **YuNet**, a lightweight, fast face detection model from OpenCV Zoo
- 🕒 Automatically logs date & time in **Indian Standard Time (IST)**
- 📝 Saves face detection logs including:
  - Roll number
  - Timestamp
  - Face count
  - Image file name
- 💾 All detected face images are stored in a dedicated folder (`face_images`)
- 📊 Logs are saved to a Google Drive CSV file (`face_log.csv`)

## 🔧 Main Hardware Requirements

Although this runs on Google Colab, the setup is suitable for integration into:
- Webcam-enabled systems (Laptop/PC with webcam)
- Raspberry Pi-based projects with camera modules
- IoT attendance systems

## 📦 Dependencies

- Python 3.x
- OpenCV
- NumPy
- Pandas
- pytz
- Google Colab modules:
  - `google.colab.output`
  - `google.colab.patches`
  - `eval_js` for webcam capture

## 📁 File Structure

```
face_recognition.py       # Main script for capturing, detecting, logging
face_images/              # Folder to store captured and processed images
face_log.csv              # CSV file with attendance/log data
```

---
