from enhanced_r3d import *
# Data Preprocessing
def format_frames(frame, output_size):
    """Format frames to tensor with specified size"""
    frame = cv2.resize(frame, output_size)
    frame = frame / 255.0
    return frame

def frames_from_video_file(video_path, n_frames=32, output_size=(224, 224), frame_step=15):
    """Extract frames from video file"""
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()

    if not ret:
        return np.zeros((n_frames, output_size[1], output_size[0], 3))

    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            # Pad with zeros if no more frames
            result.append(np.zeros_like(result[0]))

    src.release()

    result = result[:n_frames]
    while len(result) < n_frames:
        result.append(np.zeros_like(result[0]))

    result = np.array(result)
    return result

def adjust_batch_size(config, exception):
    """Adjust batch size dynamically in case of OOM."""
    if isinstance(exception, RuntimeError) and 'out of memory' in str(exception).lower():
        if config.batch_size > 1:
            config.batch_size = max(1, config.batch_size // 2)
            logging.warning(f"OOM detected. Reducing batch size to {config.batch_size}.")
            return True
    return False

class VideoDataset(Dataset):
    """Enhanced video dataset with improved frame sampling and preprocessing including advanced data augmentation."""
    
    def __init__(
        self,
        file_paths: List[str],
        targets: List[int],
        config: 'VideoClassificationConfig', 
        n_frames: int = 32,
        input_size: Tuple[int, int] = (224, 224),
        augment: bool = True
    ):
        self.file_paths = file_paths
        self.targets = targets
        self.n_frames = n_frames
        self.input_size = input_size
        self.config = config
        self.augment = augment

        if self.augment:
            self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Extract frames
            video_frames = self._extract_frames(self.file_paths[idx])

            if self.augment:
                video_frames = self._temporal_augmentation(video_frames)

            transformed_frames = torch.stack([
                self.spatial_transform(frame) for frame in video_frames
            ])  # Shape: [T, C, H, W]

            # Convert to tensor and permute to [C, T, H, W]
            video_tensor = transformed_frames.permute(1, 0, 2, 3).contiguous()

            label = self.targets[idx]

            return video_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            logging.error(f"Error processing video {self.file_paths[idx]}: {e}")
            dummy_frames = torch.zeros(3, self.n_frames, *self.input_size)
            return dummy_frames, torch.tensor(0, dtype=torch.long)

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        frames = frames_from_video_file(
            video_path,
            n_frames=self.n_frames,
            output_size=self.input_size
        )
        if len(frames) < self.n_frames:
            frames = list(frames) + [frames[-1]] * (self.n_frames - len(frames))
        return frames[:self.n_frames]

    def _temporal_augmentation(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        drop_prob = 0.2
        if random.random() < drop_prob:
            keep_indices = sorted(random.sample(range(len(frames)), k=int(len(frames) * 0.8)))
            frames = [frames[i] for i in keep_indices]

        permute_prob = 0.2
        if random.random() < permute_prob:
            start = random.randint(0, len(frames) - 5)
            end = start + random.randint(2, 5)
            subset = frames[start:end]
            random.shuffle(subset)
            frames[start:end] = subset

        if len(frames) > self.n_frames:
            frames = frames[:self.n_frames]
        elif len(frames) < self.n_frames:
            frames += [frames[-1]] * (self.n_frames - len(frames))

        return frames

def prepare_dataset(config: 'VideoClassificationConfig'):
    """Prepare dataset by collecting video file paths and labels."""
    file_paths = []
    targets = []

    dataset_root = "dataset"
    for i, cls in enumerate(config.classes):
        # Corrected the glob pattern
        search_pattern = os.path.join(dataset_root, "UCF101", "UCF-101", cls, "*.avi")
        sub_file_paths = glob.glob(search_pattern, recursive=True)[:config.videos_per_class]

        if not sub_file_paths:
            logging.warning(f"No .avi files found for class '{cls}' in '{search_pattern}'.")

        file_paths += sub_file_paths
        targets += [i] * len(sub_file_paths)

    if not file_paths:
        raise ValueError("No video files found. Please check the dataset path and file extensions.")

    # Shuffle the dataset
    combined = list(zip(file_paths, targets))
    random.shuffle(combined)

    if combined:
        file_paths, targets = zip(*combined)
        file_paths = list(file_paths)
        targets = list(targets)
    else:
        raise ValueError("No data found after shuffling.")

    train_paths, val_paths, train_targets, val_targets = train_test_split(
        file_paths, targets, test_size=0.2, random_state=42, stratify=targets
    )

    return train_paths, val_paths, train_targets, val_targets

def load_video(video_path: str, target_fps: int = 30, target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Load and preprocess video for model inference using OpenCV.
    
    Args:
        video_path (str): Path to the video file
        target_fps (int): Target frames per second
        target_size (tuple): Target frame size for resizing
    
    Returns:
        torch.Tensor: Preprocessed video tensor
    """
    try:
        # Open video using OpenCV
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate sampling rate
        sampling_rate = max(1, int(current_fps // target_fps))
        
        # Collect frames
        frames = []
        frame_count = 0
        
        while len(frames) < 16 and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sampling_rate == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame_resized = cv2.resize(frame, target_size)
                
                frames.append(frame_resized)
            
            frame_count += 1
        
        cap.release()
        
        # Pad or truncate to ensure 16 frames
        if len(frames) < 16:
            # Pad with last frame
            padding = [frames[-1]] * (16 - len(frames))
            frames.extend(padding)
        else:
            frames = frames[:16]
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
        
        # Transform frames
        video_tensor = torch.stack([transform(frame) for frame in frames])
        
        # Rearrange dimensions: [T, C, H, W] -> [C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor.unsqueeze(0)  # Add batch dimension
    
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def prepare_data(video_path, _resize, num_classes, num_frames):
    output_file = "/kaggle/working/train.txt"
    video_path = ucf11 = "/kaggle/input/realistic-action-recognition-ucf50-dataset/UCF11_updated_mpg"
    labels = [] # labels[i] = name of class with index i

    with open(output_file, "w") as f:
        x = -1
        for action_folder in os.listdir(ucf11):
            action_folder_path = os.path.join(ucf11, action_folder)
            x +=1
            if os.path.isdir(action_folder_path):
                labels.append(action_folder)
                # print(action_folder_path)
                for group_folder in os.listdir(action_folder_path):
                    group_folder_path = os.path.join(action_folder_path, group_folder)
                    if os.path.isdir(group_folder_path):
                        for video_file in os.listdir(group_folder_path):
                            video_file_path = os.path.join(group_folder_path, video_file)
                            if video_file.endswith(".mpg") or video_file.endswith(".avi"):
                                f.write(f"{video_file_path} {x}\n")
    print(labels)
    train_df = pd.read_csv("train.txt",sep = " ",header = None,names = ['path','class'])
    train_df['path'] = train_df['path'].str.replace(ucf11, '', regex=False).str.lstrip("\\")
    
    content = []
    labels = []
    
    for i in range(num_classes):
        df_temp = train_df[train_df['class'] == i]
        if not df_temp.empty:
            path = df_temp['path'].tolist()
            content.extend(path)
            labels.extend([i] * len(path))
        else:
            print(f"Class {i} không có video nào!")

    if len(content) == 0:
        raise ValueError("Không có video nào trong khoảng class được chọn!")

    content = np.array(content)
    videos = []
    y = []

    for j in range(len(content)):
        # print(f"Processing {np.round(100 * j / len(content), 3)}%: {content[j]}")
        x = video_path + '/' + content[j]
        vcap = cv2.VideoCapture(x)
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            print(f"Video {content[j]} có ít hơn {num_frames} frames. Sẽ lấy tất cả {total_frames} frames.")
            selected_indices = range(total_frames)
        else:
            selected_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        count = 0
        success = True
        while success:
            success, image = vcap.read()
            if not success:
                break
            if count in selected_indices:
                from skimage.transform import resize
                try:
                    image = resize(image, (240,320))
                    frames.append(image.astype(np.uint8))
                except Exception as e:
                    print(f"Lỗi xử lý frame {count} trong video {content[j]}: {e}")
            count += 1

        if len(frames) == num_frames:
            videos.append(np.array(frames, dtype=np.uint8))
            y.append(labels[j])
        else:
            print(f"Video {content[j]} không đủ số frame sau khi xử lý, sẽ bỏ qua.")

        vcap.release()

    videos = np.array(videos, dtype=np.uint8)
    print(f"Shape của videos: {videos.shape}")

    y = np.array(y)
    print(f"Shape của labels: {y.shape}")

    return videos, y