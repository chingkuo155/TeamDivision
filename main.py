import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import yt_dlp
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = 'yolov8_trained.pt'
model1 = YOLO(model_path)
model1.model.names = {0: 'Ball', 1: 'Hoop', 2: 'Player'}

mobilenet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
mobilenet = torch.nn.Sequential(*list(mobilenet.children())[:-1])
mobilenet.eval()

preprocess = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def download_youtube_video(url, output_path='input_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Downloaded video: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None

def extract_features(image, model, preprocess):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()

def process_frame(frame):
    results = model1.predict(frame)

    features_list = []
    for result in results[0].boxes:
        class_id = int(result.cls[0])
        if class_id == 2:  # 只處理球員
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cropped_img = frame[y1:y2, x1:x2]
            cropped_pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            features = extract_features(cropped_pil_img, mobilenet, preprocess)
            features_list.append(features)

    n = len(features_list)
    if n < 2:
        return frame

    features_2d = np.array(features_list)
    scaler = StandardScaler()
    features_2d = scaler.fit_transform(features_2d)

    dbscan = DBSCAN(eps=0.05, min_samples=1).fit(features_2d)
    labels = dbscan.labels_

    colors = [(255, 0, 0), (0, 0, 255)]  # 紅色代表B隊，藍色代表A隊
    color_mapping = {}

    for i, label in enumerate(labels):
        if label not in color_mapping:
            color_mapping[label] = colors[len(color_mapping) % len(colors)]

    j = 0
    for i, result in enumerate(results[0].boxes):
        if j == n:
            break
        class_id = int(result.cls[0])
        if class_id == 2:
            label = labels[j]
            if label == -1:
                continue  # 忽略噪點

            x1, y1, x2, y2 = map(int, result.xyxy[0])
            score = result.conf[0]
            team_label = '(A)' if color_mapping[label] == colors[1] else '(B)'
            color = color_mapping[label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Player {team_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f'{score:.2f}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            j += 1

    return frame

def process_video(input_path, output_path, target_frames=60, duration=60):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = target_frames / duration

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    total_frames = int(original_fps * duration)
    skip_rate = max(1, total_frames // target_frames)

    frame_count = 0
    processed_count = 0

    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_rate == 0:
            processed_frame = process_frame(frame)
            out.write(processed_frame)
            processed_count += 1

        frame_count += 1

    cap.release()
    out.release()
    return output_path

@app.post("/upload/")
async def upload_file(file: UploadFile):
    input_path = "input_video.mp4"
    output_path = "output_video.mp4"
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    process_video(input_path, output_path, target_frames=60, duration=60)
    return FileResponse(output_path)

@app.post("/process_youtube/")
async def process_youtube(url: str = Form(...)):
    input_path = download_youtube_video(url)
    if input_path:
        output_path = "output_video.mp4"
        process_video(input_path, output_path, target_frames=60, duration=60)
        return FileResponse(output_path)
    return {"error": "Failed to download video"}

@app.get("/")
async def main():
    return RedirectResponse(url="/static/index.html")