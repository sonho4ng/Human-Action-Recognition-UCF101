import pandas as pd
import os
import cv2
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset

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


class UCF11Dataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data # numpy array or PIL Image
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]

        if self.transform:
            data = self.transform(data)
    
        return data, target