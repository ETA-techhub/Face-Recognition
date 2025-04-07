from IPython import get_ipython
from IPython.display import display
# %%
import cv2
import numpy as np
import pandas as pd
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import urllib.request
import os
from datetime import datetime
import pytz
from google.colab import drive
drive.mount('/content/drive') # Removed the extra indentation here

# ✅ Set IST (Indian Standard Time)
LOCAL_TIMEZONE = "Asia/Kolkata"

# 📥 Download YuNet model (only needed once)
yunet_model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
model_path = "yunet.onnx"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(yunet_model_url, model_path)
    print("✅ YuNet model downloaded!")

# 🖼️ Load YuNet face detector
face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))

# 📂 Create folder for saving images
SAVE_FOLDER = "face_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

!chmod 777 face_log.csv

# 📜 Log file for storing records
LOG_FILE ='/content/drive/MyDrive/face_log.csv'

# ⏰ Function to get local IST time
def get_local_time():
    utc_now = datetime.utcnow()
    local_tz = pytz.timezone(LOCAL_TIMEZONE)
    local_time = utc_now.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_time

# 📸 Function to take a photo
def take_photo(roll_number):
    js = Javascript('''
        async function takePhoto() {
            const div = document.createElement('div');
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');
            const button = document.createElement('button');
            button.textContent = 'Take Photo';
            document.body.appendChild(div);
            div.appendChild(video);
            div.appendChild(button);

            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();

            await new Promise((resolve) => button.onclick = resolve);

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getTracks().forEach(track => track.stop());
            div.remove();

            return canvas.toDataURL('image/jpeg');
        }
    ''')

    display(js)
    data = eval_js('takePhoto()')
    binary = b64decode(data.split(',')[1])

    # ⏰ Convert to IST time
    local_time = get_local_time()

    # ✅ Filename format: MM_DD_YYYY__HH:MM
    timestamp_filename = local_time.strftime("%m_%d_%Y__%I:%M")
    readable_date = local_time.strftime("%B %d, %Y - %I:%M %p IST")

    image_path = os.path.join(SAVE_FOLDER, f"{roll_number}_{timestamp_filename}.jpg")

    with open(image_path, 'wb') as f:
        f.write(binary)

    return image_path, readable_date, timestamp_filename

# 🎭 Function to detect faces and add text overlay
def detect_faces(image_path, roll_number, readable_date):
    img = cv2.imread(image_path)

    # Resize image for YuNet input
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))

    # 🧐 Detect faces
    faces = face_detector.detect(img)

    face_count = 0
    if faces[1] is not None:
        face_count = len(faces[1])
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])  # Bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # ✏️ Add Roll Number, Date & Time in the Bottom Left Corner
    text = f"Roll: {roll_number} | {readable_date}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 255)  # Yellow text
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_x = 10
    text_y = height - 10  # Bottom-left corner
    cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    return img, face_count

# 📝 Function to log data
def log_data(roll_number, readable_date, face_count, filename):
    data = {
        "Roll Number": [roll_number],
        "Date & Time": [readable_date],
        "Face Count": [face_count],
        "Image File": [filename]
    }

    df = pd.DataFrame(data)

    # Append to existing CSV file if it exists
    if os.path.exists(LOG_FILE):
        existing_df = pd.read_csv(LOG_FILE)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(LOG_FILE, index=False)
    print(f"✅ Log saved to {LOG_FILE}")

# 🚀 Main Execution: Capture & Process Image
try:
    roll_number = input("Enter Roll Number: ").strip()
    if not roll_number:
        raise ValueError("❌ Roll Number cannot be empty!")

    # 📸 Capture Image
    filename, readable_date, timestamp_filename = take_photo(roll_number)
    print(f"✅ Photo saved as: {filename}")

    # 🎭 Detect Faces & Add Text Overlay
    result_img, face_count = detect_faces(filename, roll_number, readable_date)

    # Save detected image with "_detected"
    detected_filename = filename.replace(".jpg", "_detected.jpg")
    cv2.imwrite(detected_filename, result_img)

    # ✅ Log data
    log_data(roll_number, readable_date, face_count, detected_filename)

    # Display processed image
    cv2_imshow(result_img)

    # 📥 Provide download option
    # files.download(detected_filename) # This line needs to be addressed in a separate issue.

    # 🔔 Detection result
    if face_count > 0:
        print(f"✅ {face_count} face(s) detected.")
    else:
        print("⚠️ No faces detected. Try better lighting or adjust angle.")

except Exception as e:
    print("❌ Error:", str(e))