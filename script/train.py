import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import os
import sys

from dataset.UCF101 import VideoDataset
from models.resnet_lstm import ResNetLSTM
from models.ResidualSE import ResidualSE
from models.tsm import models
from models.i3d_shufflenet import EnhancedI3DShuffleNet
from models.enhanced_r3d import R3DModel


import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description="Training parameters")

parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--device", choices=['cuda', 'cpu'], default='cuda', help="Choose device: 'cuda' or 'cpu'")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument("--videos_per_class", type=int, default=50, help="Number of videos per class")
parser.add_argument("--n_frames", type=int, default=10, help="Number of frames per video")
parser.add_argument("--model", choices=['resnet-lstm', 'residualSE', 'tsm', 'i3d', 'enhanced_r3d'], default='resnet-lstm', help="Choose models: ")
parser.add_argument("--dataset", choices=['ucf101', 'ucf11'], default='ucf101', help="Choose datasets: ucf101 or ucf11")
parser.add_argument("--dataset_path", type=str, default='/Users/nguyentrithanh/it3320e-human-action-recognition-ucf101/data', help="Path to the folder that contain the dataset: <Path>/UCF101/train/... ")
parser.add_argument("--use_wandb", choices=['True', 'False'], default='False', help="Choosing to use wandb or not")

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

def check_path(path: str) -> None:
    if os.path.exists(path):
        print(f"Checkpoint path: {path}")
    else:
        print("Checkpoint file does not exist!")

def choose_model(model_name: str) -> object:
    if model_name == 'resnet-lstm':
        model = ResNetLSTM(num_classes=CFG.num_classes).to(CFG.device)
    elif model_name == 'residualSE':
        model = ResidualSE(num_classes=CFG.num_classes).to(CFG.device)
    elif model_name == 'tsm':
        model = models.TSN(num_classes=CFG.num_classes, num_segments=CFG.n_frames, modality="RGB",
                   base_model='resnet50', before_softmax=False,
                   is_shift=True, shift_place='blockres').to(CFG.device)
    elif model_name == 'i3d':
        model = EnhancedI3DShuffleNet(CFG.num_classes)
    elif model_name == 'enhanced_r3d':
        model = R3DModel(num_classes=CFG.num_classes, pretrained='update_r3d', dropout_prob=0.5)
    return model

class CFG:
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    learning_rate = args.learning_rate
    num_workers = args.num_workers
    videos_per_class = args.videos_per_class
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
    n_frames = args.n_frames
    model_name = args.model
    dataset_name = args.dataset
    if dataset_name == "ucf101":
        num_classes = 101
    elif dataset_name == "ucf11":
        num_classes = 11

    checkpoint_path = os.path.abspath(os.path.join(script_dir, f"../checkpoint/{model_name}.pth"))
    check_path(checkpoint_path)
    
    dataset_path = args.dataset_path
    use_wandb = args.use_wandb

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(dataloader):
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(dataloader):
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc, all_preds, all_labels

def main():
    if CFG.use_wandb == "True":
        wandb.login(key=os.environ["WANDB"])
        wandb.init(
            project="DL Score",
            config={
                "batch_size": CFG.batch_size,
                "learning_rate": CFG.learning_rate,
                "epochs": CFG.epochs,
                "clip_duration": CFG.n_frames,
                "model": CFG.model_name,
                "num_classes": CFG.num_classes,
            }
        )
    

    # Load dataset
    file_paths = []
    targets = []
    for i, cls in enumerate(CFG.classes):
        sub_file_paths = glob.glob(f"{CFG.dataset_path}/UCF101/train/{cls}/**.avi")[:CFG.videos_per_class]  #replace the path to your local dataset
        file_paths += sub_file_paths
        targets += [i] * len(sub_file_paths)

    # Split dataset
    train_paths, val_paths, train_targets, val_targets = train_test_split(
        file_paths, targets, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = VideoDataset(train_paths, train_targets)
    val_dataset = VideoDataset(val_paths, val_targets)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers
    )

    # Initialize model
    model = choose_model(CFG.model_name)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(CFG.epochs):
        print(f'Epoch {epoch+1}/{CFG.epochs}')
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CFG.device
        )
        
        val_loss, val_acc, all_preds, all_labels = validate(
            model, val_loader, criterion, CFG.device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        if CFG.use_wandb == 'True':
            wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Learning Rate": optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CFG.checkpoint_path

            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            # 'val_accuracy': val_acc
            }, checkpoint_path)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')


    return model, history


if __name__ == "__main__":
    model, history = main()


    # class CFG:
    #     epochs = 200
    #     batch_size = 8
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     learning_rate = 0.00005
    #     num_workers = 2
    #     classes = [
    #         "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    #         "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    #         "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    #         "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    #         "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    #         "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    #         "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "HammerThrow",
    #         "Hammering", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    #         "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    #         "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
    #         "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    #         "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    #         "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    #         "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    #         "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    #         "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    #         "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    #         "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    #         "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    #         "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    #         "YoYo"
    #     ]
    #     videos_per_class = 50
