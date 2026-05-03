import os
import torch
from torch.utils.data import Dataset, DataLoader

class EpicI3DDataset(Dataset):
    def __init__(self, list_file, data_dir, num_segments=16, debug=False):
        """
        list_file: path to .txt file listing features and labels
        num_segments: number of consecutive frames per sequence
        """
        self.debug = debug
        self.num_segments = num_segments
        self.samples = []
        # parse the txt file
        with open(list_file, "r") as f:
            for line in f:
                dir_path, frame_idx, label = line.strip().split()
                dir_path = os.path.join(data_dir, dir_path)
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

        half_window = int(self.num_segments // 2 - 1) # should be 7
        total_window = self.num_segments  # should be 16
        zero_count = 0
        loaded_count = 0
        feat_list = []
        
        for i in range(frame_idx - half_window, frame_idx + (total_window - half_window)):
            if i < 1:
            # pad with zeros for frames before the start
                feat = torch.zeros(2048)
            else:
                fname = f"img_{i:05d}.t7"
                feat_path = os.path.join(dir_path, fname)
                if os.path.exists(feat_path):
                    try:
                        feat = torch.load(feat_path, map_location="cpu").float()
                        loaded_count += 1
                    except Exception as e:
                        if self.debug:
                            print(f"Error loading {feat_path}: {e}")
                        feat = torch.zeros(2048)
                        zero_count += 1
                else:
                    # pad with zeros if frame does not exist (end of video)
                    feat = torch.zeros(2048)
                    zero_count += 1
            feat_list.append(feat)

        features = torch.stack(feat_list, dim=0)  # [num_segments, 2048]

        if self.debug and idx < 2:  # Only print for first 2 samples
            print(f"\n[Sample {idx}] Label: {label}")
            print(f"  - Loaded frames: {loaded_count}, Padded frames: {zero_count}")
            print(f"  - Features shape: {features.shape}")
            print(f"  - Features mean: {features.mean():.6f}, std: {features.std():.6f}")
            print(f"  - Features min: {features.min():.6f}, max: {features.max():.6f}")
            print(f"  - Non-zero frames: {(features.abs().sum(dim=1) > 0).sum()}/{self.num_segments}")

        return features, label

def pad_collate(batch):
    # pad_collate function to batch the frame sequences
    feats, labels = zip(*batch)
    feats = torch.stack(feats, dim=0)  # [B, num_segments, 2048]
    labels = torch.tensor(labels, dtype=torch.long)
    return feats, labels

def get_epic_dloader(train_list, test_list, data_dir='data', batch_size=32, num_segments=16, num_workers=4, debug=False):
    train_dataset = EpicI3DDataset(train_list, data_dir, num_segments, debug=debug)
    test_dataset  = EpicI3DDataset(test_list, data_dir, num_segments, debug=debug)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers,
                                            collate_fn=pad_collate, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers,
                                            collate_fn=pad_collate, pin_memory=True)
    
    return train_loader, test_loader

def debug_features(loader, num_batches=2):
    """Inspect feature statistics across batches"""
    print("\n" + "="*60)
    print("FEATURE STATISTICS DEBUG")
    print("="*60)
    
    for batch_idx, (feats, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        print(f"\n[Batch {batch_idx}]")
        print(f"  - Shape: {feats.shape}")
        print(f"  - Batch labels: {labels}")
        print(f"  - Batch mean: {feats.mean():.8f}")
        print(f"  - Batch std: {feats.std():.8f}")
        print(f"  - Batch min: {feats.min():.8f}, max: {feats.max():.8f}")
        print(f"  - Percentage of zero features: {(feats.abs().sum(dim=(1,2)) == 0).float().mean()*100:.2f}%")
        
        # Per-sample stats
        for sample_idx in range(min(3, feats.shape[0])):
            sample = feats[sample_idx]  # [num_segments, 2048]
            non_zero_frames = (sample.abs().sum(dim=1) > 0).sum().item()
            print(f"    Sample {sample_idx}: {non_zero_frames}/{sample.shape[0]} frames loaded, "
                  f"mean={sample.mean():.8f}, std={sample.std():.8f}")


if __name__ == "__main__":
    # to test the output of pre-extracted features
    train_list = "data/frame_annotations_transVAE/list_P01_train.txt"
    test_list  = "data/frame_annotations_transVAE/list_P01_test.txt"
    num_segments = 16
    data_dir = 'data'

    print("Creating dataloaders with debug mode...")
    train_loader, test_loader = get_epic_dloader(
        train_list, test_list, data_dir=data_dir, 
        batch_size=50, num_segments=16, num_workers=0, debug=True
    )
    
    print("\n\nTRAIN LOADER:")
    debug_features(train_loader, num_batches=3)
    
    print("\n\nTEST LOADER:")
    debug_features(test_loader, num_batches=3)

    # train_dataset = EpicI3DDataset(train_list, num_segments)
    # test_dataset  = EpicI3DDataset(test_list, num_segments)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
    #                                         shuffle=True, num_workers=1,
    #                                         collate_fn=pad_collate)
    # test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32,
    #                                         shuffle=False, num_workers=1,
    #                                         collate_fn=pad_collate)

    # # Example batch 
    # for feats, labels in train_loader:
    #     print(feats.shape)   # [B, num_segments, D=2048]
    #     print(labels.shape)  # [B]
    #     break
