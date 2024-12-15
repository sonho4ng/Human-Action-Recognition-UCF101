import os
import logging
from typing import List, Tuple, Dict, Optional
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import cv2
from tqdm import tqdm
import gc
import ipywidgets as widgets
from IPython.display import display
from preprocess import load_video

gc.collect()
torch.cuda.empty_cache()


# Use your own wandb key you want to use wandb for logging
# WANDB_API_KEY = ''
# wandb.login(key=WANDB_API_KEY)


# Configuration
class VideoClassificationConfig:
    """Enhanced configuration for video classification with R3D and ShuffleNet."""
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.0001,
        num_workers: int = 4,
        videos_per_class: int = 60,
        model_type: str = 'update_r3d',
        pretrained: bool = True,
        accumulation_steps: int = 1,  # For gradient accumulation
        use_amp: bool = True,
        max_batch_size: int = 32,  # Maximum batch size to attempt
        wandb_project: str = 'har',
        checkpoint_path: str = 'results/update_r3d.pth',
        resume: bool = True,
        scheduler_mode: str = 'plateau',  # 'plateau' or 'cosine'
        scheduler_factor: float = 0.1,  # Factor for ReduceLROnPlateau
        scheduler_patience: int = 5,  # Patience for ReduceLROnPlateau
        early_stop_patience: int = 10,  # Patience for Early Stopping
        checkpoint_interval: int = 10,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.videos_per_class = videos_per_class
        self.model_type = model_type
        self.pretrained = pretrained
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.max_batch_size = max_batch_size
        self.wandb_project = wandb_project
        self.checkpoint_path = checkpoint_path
        self.resume = resume
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.early_stop_patience = early_stop_patience
        self.checkpoint_interval = checkpoint_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classes = [
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

def inference_test(
    model: torch.nn.Module, 
    video_paths: List[str], 
    config: VideoClassificationConfig
) -> None:
    """
    Perform inference on a list of video paths.
    
    Args:
        model (torch.nn.Module): Trained classification model
        video_paths (List[str]): List of video file paths to test
        config (VideoClassificationConfig): Configuration object
    """
    model.eval()
    
    with torch.no_grad():
        for video_path in video_paths:
            print(f"\nProcessing video: {video_path}")
            
            video_tensor = load_video(video_path)
            
            if video_tensor is None:
                print(f"Skipping {video_path} due to loading error")
                continue
            
            video_tensor = video_tensor.to(config.device)
            
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            
            predicted_class = config.classes[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
            
            # Print top-3 predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=3, dim=1)
            print("\nTop 3 Predictions:")
            for k in range(3):
                class_idx = top_k_indices[0][k].item()
                class_name = config.classes[class_idx]
                prob = top_k_probs[0][k].item()
                print(f"{k+1}. {class_name}: {prob * 100:.2f}%")

# Modelling
class R3DModel(nn.Module):
    """R3D model for video classification."""
    def __init__(self, num_classes: int, pretrained: bool = True, dropout_prob: float = 0.5):
        super(R3DModel, self).__init__()
        self.model = video_models.r3d_18(weights='KINETICS400_V1' if pretrained else None)
        
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
class AttentionBlock(nn.Module):
    """Spatial-Temporal Attention Mechanism"""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attn = self.channel_attention(x)
        spatial_attn = self.spatial_attention(x)
        return x * channel_attn * spatial_attn

class EnhancedR3DModel(nn.Module):
    """Update R3D model with dynamic feature handling"""
    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True, 
        dropout_prob: float = 0.5
    ):
        super().__init__()
        
        # Base model initialization
        self.base_model = video_models.r3d_18(weights='KINETICS400_V1' if pretrained else None)
        
        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Create a feature dimension calculator
        self._calculate_feature_dimensions()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob / 2),
            nn.Linear(512, num_classes)
        )

    def _calculate_feature_dimensions(self):
        """Dynamically calculate feature dimensions"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 16, 224, 224)
            features = self.features(dummy_input)
            
            # Flatten the features
            self.feature_dim = features.view(features.size(0), -1).size(1)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        # Extract features
        features = self.features(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output    

def create_model(config: VideoClassificationConfig) -> nn.Module:
    """Factory method to create appropriate video classification model."""
    num_classes = len(config.classes)

    if config.model_type == 'r3d':
        model = R3DModel(
            num_classes=num_classes,
            pretrained=config.pretrained,
            dropout_prob=0.5
        )

    if config.model_type == 'update_r3d':
        model = EnhancedR3DModel(
            num_classes=num_classes,
            pretrained=config.pretrained,
            dropout_prob=0.5
        ) 

    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    model = model.to(config.device)
    return model

# Training
def train_epoch(model, dataloader, criterion, optimizer, device, scaler, config):
    """Train model for one epoch with enhanced logging and progress tracking."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batches = len(dataloader)

    # Progress bar for the entire epoch
    progress_bar = tqdm(enumerate(dataloader), total=batches, desc="Training")

    optimizer.zero_grad()

    for batch_idx, (videos, labels) in progress_bar:
        videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda',enabled=config.use_amp):
            outputs = model(videos)
            loss = criterion(outputs, labels) / config.accumulation_steps

        # Gradient accumulation
        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # Compute batch statistics
        running_loss += loss.item() * config.accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Compute and update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total

        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Accuracy': f'{current_acc:.2f}%'
        })

        # Log to Weights & Biases every batch
        wandb.log({
            'Train/Loss': current_loss,
            'Train/Accuracy': current_acc,
            'Train/Batch': batch_idx + 1
        })

    # Compute epoch-level metrics
    epoch_loss = running_loss / batches
    epoch_acc = 100. * correct / total

    logging.info(f"\nTraining Epoch Summary:")
    logging.info(f"Epoch Loss: {epoch_loss:.4f}")
    logging.info(f"Epoch Accuracy: {epoch_acc:.2f}%")

    # Log epoch metrics to Weights & Biases
    wandb.log({
        'Train/Epoch Loss': epoch_loss,
        'Train/Epoch Accuracy': epoch_acc
    })

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, config):
    """Validate model performance with enhanced logging and tracking."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    batches = len(dataloader)
    all_preds = []
    all_labels = []

    # Progress bar for validation
    progress_bar = tqdm(dataloader, total=batches, desc="Validation")

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(progress_bar):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=config.use_amp):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            # Compute batch statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Collect predictions for final analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Compute and update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total

            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Accuracy': f'{current_acc:.2f}%'
            })

            # Log to Weights & Biases every batch
            wandb.log({
                'Validation/Loss': current_loss,
                'Validation/Accuracy': current_acc,
                'Validation/Batch': batch_idx + 1
            })

    # Compute validation-level metrics
    val_loss = running_loss / batches
    val_acc = 100. * correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='weighted',
        zero_division=0
    )

    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=config.classes, 
        zero_division=0
    )

    logging.info(f"\nValidation Summary:")
    logging.info(f"Validation Loss: {val_loss:.4f}")
    logging.info(f"Validation Accuracy: {val_acc:.2f}%")
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    wandb.log({
        'Validation/Epoch Loss': val_loss,
        'Validation/Epoch Accuracy': val_acc,
        'Validation/Precision': precision,
        'Validation/Recall': recall,
        'Validation/F1-Score': f1
    })

    return val_loss, val_acc, all_preds, all_labels, class_report

def load_checkpoint(config, model, optimizer, scheduler, scaler):
    """Load checkpoint if available."""
    if os.path.isfile(config.checkpoint_path):
        logging.info(f"Loading checkpoint from '{config.checkpoint_path}'")
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0)
        logging.info(f"Loaded checkpoint '{config.checkpoint_path}' (Epoch {start_epoch})")
        return start_epoch, best_val_acc
    else:
        logging.info(f"No checkpoint found at '{config.checkpoint_path}'. Starting fresh.")
        return 0, 0

def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_custom(y_true, y_pred, class_names):
    """Generate and plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Execution (Training)
config = VideoClassificationConfig(
    model_type='update_r3d',
    epochs=66,
    batch_size=16,
    learning_rate=1e-4,
    accumulation_steps=2,
    use_amp=True,
    wandb_project='thanhnx_har',
    checkpoint_path='results/update_r3d.pth',
    resume=True,
    scheduler_mode='plateu',
    scheduler_factor=0.1,
    scheduler_patience=4,
    early_stop_patience=10,
    checkpoint_interval=10
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def train_r3d():
    wandb.init(
        project=config.wandb_project,
        config=vars(config),
        resume='allow' if config.resume else False,
        job_type='training',
        settings=wandb.Settings(init_timeout=120)
    )

    # Prepare dataset
    train_paths, val_paths, train_targets, val_targets = prepare_dataset(config)
    logging.info(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
    wandb.config.update({
        'train_samples': len(train_paths),
        'val_samples': len(val_paths)
    })

    # Create datasets
    train_dataset = VideoDataset(train_paths, train_targets, config)
    val_dataset = VideoDataset(val_paths, val_targets, config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda',enabled=config.use_amp)

    start_epoch = 0
    best_val_acc = 0
    if config.resume:
        start_epoch, best_val_acc = load_checkpoint(config, model, optimizer, scheduler, scaler)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(start_epoch, config.epochs):
        if early_stop:
            logging.info("Early stopping triggered.")
            break

        print(f'Epoch {epoch+1}/{config.epochs}')
        logging.info(f'Epoch {epoch+1}/{config.epochs}')
        wandb.log({'epoch': epoch + 1})

        try:
            # Training phase
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, config.device, scaler, config
            )

            # Validation phase
            val_loss, val_acc, all_preds, all_labels, class_report = validate(
                model, val_loader, criterion, config.device, config
            )

            # Step the scheduler with validation accuracy
            scheduler.step(val_acc)

            # Retrieve and log the current learning rate
            current_lr = get_current_lr(optimizer)
            wandb.log({'Learning Rate': current_lr})
            logging.info(f"Current Learning Rate: {current_lr}")

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, config.checkpoint_path)
                logging.info(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

                # Create a W&B Artifact
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file(config.checkpoint_path)
                wandb.log_artifact(artifact)
                logging.info(f"Checkpoint {config.checkpoint_path} logged to W&B Artifacts.")
                print(f"Checkpoint {config.checkpoint_path} logged to W&B Artifacts.")

            else:
                epochs_no_improve += 1
                logging.info(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

            # Save checkpoint every N epochs
            if (epoch + 1) % config.checkpoint_interval == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, checkpoint_name)
                logging.info(f"Checkpoint saved at epoch {epoch+1} as '{checkpoint_name}'.")

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            logging.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logging.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Check early stopping condition
            if epochs_no_improve >= config.early_stop_patience:
                logging.info(f"No improvement for {config.early_stop_patience} consecutive epochs. Stopping training.")
                early_stop = True

        except RuntimeError as e:
            if adjust_batch_size(config, e):
                # Reinitialize data loaders with new batch size
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                    pin_memory=True,
                    persistent_workers=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True,
                    persistent_workers=True
                )

                # Reinitialize optimizer and scheduler
                optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=config.scheduler_factor,
                    patience=config.scheduler_patience,
                    min_lr=1e-6
                )
                scaler = torch.amp.GradScaler(enabled=config.use_amp)

                # Retry the current epoch
                epoch -= 1
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                logging.error("Unrecoverable CUDA OOM error.")
                raise e

        # Clean up to free memory
        torch.cuda.empty_cache()
        gc.collect()

    wandb.finish()

    # Plot training history
    plot_training_history(history)

    # Plot confusion matrix for final epoch
    plot_confusion_matrix_custom(all_labels, all_preds, config.classes)

    # Optionally, print classification report
    print("\nClassification Report:")
    print(class_report)

    # Save the final model
    final_model_path = 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to '{final_model_path}'.")

def inference_test(
    model: torch.nn.Module, 
    video_paths: List[str], 
    config: VideoClassificationConfig
) -> None:
    """
    Perform inference on a list of video paths.
    
    Args:
        model (torch.nn.Module): Trained classification model
        video_paths (List[str]): List of video file paths to test
        config (VideoClassificationConfig): Configuration object
    """
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for video_path in video_paths:
            print(f"\nProcessing video: {video_path}")
            
            # Load and preprocess video
            video_tensor = load_video(video_path)
            
            if video_tensor is None:
                print(f"Skipping {video_path} due to loading error")
                continue
            
            # Move tensor to appropriate device
            video_tensor = video_tensor.to(config.device)
            
            # Perform inference
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            
            # Get predicted class name
            predicted_class = config.classes[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence * 100:.2f}%")
            
            # Print top-3 predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=3, dim=1)
            print("\nTop 3 Predictions:")
            for k in range(3):
                class_idx = top_k_indices[0][k].item()
                class_name = config.classes[class_idx]
                prob = top_k_probs[0][k].item()
                print(f"{k+1}. {class_name}: {prob * 100:.2f}%")