import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import sys
import os

from dataset.UCF101 import VideoDataset
from models.resnet_lstm import ResNetLSTM

script_dir = os.path.dirname(os.path.abspath(__file__))
resnet_lstm_path = os.path.abspath(os.path.join(script_dir, "../checkpoint/resnet-lstm.pth"))

if os.path.exists(resnet_lstm_path):
    print(f"Checkpoint path: {resnet_lstm_path}")
else:
    print("Checkpoint file does not exist!")


import argparse

parser = argparse.ArgumentParser(description="Inferred Video Path")

parser.add_argument("--infer_path", type=str, required=True)
args = parser.parse_args()
infer_path = args.infer_path


class CFG:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def main():
    # Load dataset
    file_paths = [infer_path]
    targets = [0.0]
    # for i, cls in enumerate(CFG.classes):
    #     sub_file_paths = glob.glob(f"/kaggle/input/ucf101-action-recognition/train/{cls}/**.avi")[-CFG.videos_per_class-1:-1] 
    #     file_paths += sub_file_paths
    #     targets += [i] * len(sub_file_paths)
    
    # Create datasets and dataloaders
    infer_dataset = VideoDataset(file_paths, targets)
    infer_loader = DataLoader(infer_dataset, batch_size=1)


    # Initialize model
    model = ResNetLSTM(num_classes=len(CFG.classes)).to(CFG.device)
    checkpoint = torch.load(resnet_lstm_path, weights_only=True, map_location=CFG.device)
    model.load_state_dict(checkpoint['model_state_dict'])  

    model.eval()

    all_preds = []

    with torch.no_grad():
        for videos, labels in tqdm(infer_loader):
            videos, labels = videos.to(CFG.device), labels.to(CFG.device)
            
            outputs = model(videos)

            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            
            # Apply softmax to get class probabilities
            probabilities = F.softmax(outputs, dim=1)  # Softmax over the class dimension (dim=1)

            # Optionally, get the predicted class with the highest probability
            predicted_class = torch.argmax(probabilities)
            print(f"Predicted Class: {CFG.classes[int(predicted_class.item())]}")

            
            # Get the class probabilities as a numpy array
            probabilities = probabilities.cpu()
            
            top5_probs, top5_classes = torch.topk(probabilities, 5)  # Get top 5 classes
            
            # Convert probabilities and class indices to numpy for easy visualization
            top5_probs = top5_probs.squeeze().numpy()
            top5_classes = top5_classes.squeeze().numpy()
            
            # Print top 5 class probabilities
            print("Top 5 Class Probabilities:")
            for i in range(5):
                print(f"Class {CFG.classes[top5_classes[i]]}: {top5_probs[i] * 100:.2f}%")
    
            print("#######################################")


if __name__ == "__main__":
    main()