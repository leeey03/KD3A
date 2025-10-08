# dataloader should output clips instead of images

## test viewing of the data annotations
# import pickle
# import pandas as pd

# file_path = "data/D1_test.pkl"
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
# print(data.head())
# print(data['verb_class'].value_counts())

# checking the I3D feature dimensions 
# import torch
# data = torch.load('data/I3D-feature-pretrain/train/P01/P01_01/0/img_00002.t7')
# print(data.shape)

import os
import torch
from torch.utils.data import Dataset, DataLoader

class EpicI3DDataset(Dataset):
    def __init__(self, list_file, num_segments=16):
        """
        list_file: path to .txt file listing features and labels
        num_segments: number of consecutive frames per sequence
        """
        self.num_segments = num_segments
        self.samples = []
        # parse the txt file
        with open(list_file, "r") as f:
            for line in f:
                dir_path, frame_idx, label = line.strip().split()
                frame_idx = int(frame_idx)
                label = int(label)

                # store the start frame index and label
                self.samples.append((dir_path, frame_idx, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dir_path, frame_idx, label = self.samples[idx]
        # Similar to TranSVAE, a temporal window is constructed for each frame using previous 7 frames, the current frame and the next 8 frames
        # zero padding is used for the start and end of the video

        half_window = int(self.num_segments - 1) # should be 7
        total_window = self.num_segments  # should be 16

        feat_list = []
        for i in range(frame_idx - half_window, frame_idx + (total_window - half_window)):
            if i < 1:
            # pad with zeros for frames before the start
                feat = torch.zeros(2048)
            else:
                fname = f"img_{i:05d}.t7"
                feat_path = os.path.join(dir_path, fname)
            if os.path.exists(feat_path):
                feat = torch.load(feat_path, map_location="cpu").float()
            else:
                # pad with zeros if frame does not exist (end of video)
                feat = torch.zeros(2048)
            feat_list.append(feat)

        features = torch.stack(feat_list, dim=0)  # [num_segments, 2048]
        return features, label

def pad_collate(batch):
    # pad_collate function to batch the frame sequences
    feats, labels = zip(*batch)
    feats = torch.stack(feats, dim=0)  # [B, num_segments, 2048]
    labels = torch.tensor(labels, dtype=torch.long)
    return feats, labels

def get_epic_dloader(train_list, test_list, batch_size=32, num_segments=16, num_workers=4):
    train_dataset = EpicI3DDataset(train_list, num_segments)
    test_dataset  = EpicI3DDataset(test_list, num_segments)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            collate_fn=pad_collate)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            collate_fn=pad_collate)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_list = "data/frame_annotations_transVAE/list_P01_train.txt"
    test_list  = "data/frame_annotations_transVAE/list_P01_test.txt"
    num_segments = 16

    train_dataset = EpicI3DDataset(train_list, num_segments)
    test_dataset  = EpicI3DDataset(test_list, num_segments)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                            shuffle=True, num_workers=1,
                                            collate_fn=pad_collate)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                            shuffle=False, num_workers=1,
                                            collate_fn=pad_collate)

    # Example batch 
    for feats, labels in train_loader:
        print(feats.shape)   # [B, num_segments, D=2048]
        print(labels.shape)  # [B]
        break
