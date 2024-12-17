import cv2
import torch
import numpy as np
import streamlit as st
from torchvision.transforms import Compose, ToTensor
from models.resnet_lstm import ResNetLSTM

# Số lớp và thông số
NUM_CLASSES = 101
SEQ_LENGTH = 10
FRAME_SIZE = (224, 224)

import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context
classes = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "HammerThrow",
    "Hammering", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo"
]
# Kiểm tra thiết bị (CUDA hoặc CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#st.write(f"Using device: {device}")

# Tải mô hình
@st.cache_resource
def load_model():
    model = ResNetLSTM(num_classes=NUM_CLASSES)
    checkpoint = torch.load('checkpoint/resnet-lstm.pth', weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  
    model.to(device)
    model.eval()
    return model

model = load_model()

# Hàm tiền xử lý frame
def preprocess_frame(frame):
    transform = Compose([ToTensor()])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, FRAME_SIZE)
    tensor = transform(frame).to(device)
    return tensor

# Hàm dự đoán
def predict_from_buffer(buffer):
    input_tensor = torch.stack(buffer, dim=1).unsqueeze(0).permute(0,2,1,3,4).to(device)  # (1, C, seq, H, W)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Streamlit UI
st.title("Real-Time Video Prediction fro Human Action Recognition")
st.write("Upload a video to predict class in real-time.")

# Upload video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mpg"])
if uploaded_file is not None:
    # Lưu file tạm thời
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_video_path)  # Hiển thị video gốc

    # Đọc video
    cap = cv2.VideoCapture(temp_video_path)
    frame_buffer = []
    frame_placeholder = st.empty()  # Tạo vị trí cố định để hiển thị frame
    text_placeholder = st.empty()  # Tạo vị trí cố định để hiển thị text dự đoán

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tiền xử lý frame
        processed_frame = preprocess_frame(frame)
        frame_buffer.append(processed_frame)

        # Khi buffer đủ 16 frame, thực hiện dự đoán
        if len(frame_buffer) == SEQ_LENGTH:
            # frame_buffer = frame_buffer.permute(1, 0, 2, 3)
            # frame_buffer = frame_buffer.unsqueeze(0)
            predicted_class = predict_from_buffer(frame_buffer)
            frame_buffer.pop(0)

            # Hiển thị lớp dự đoán trên frame
            classe = classes[predicted_class]
            label = f"{classe}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Cập nhật frame và text ở vị trí cố định
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            text_placeholder.write(label)

    cap.release()
