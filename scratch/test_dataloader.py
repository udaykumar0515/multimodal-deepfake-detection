import torch
from dataset import create_dataloaders
from models import MultiHeadDeepfakeModel

def main():
    print("Testing V2 DataLoader Implementation...")
    
    csv_dir = "data/dataset_split"
    video_dir = "data/processed_frames"
    audio_dir = "data/processed_audio/spectrograms"
    
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_dir=csv_dir,
        video_dir=video_dir,
        audio_dir=audio_dir,
        batch_size=2,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch from Train Loader
    for batch in train_loader:
        assert len(batch) == 5, f"Expected 5 items in batch, got {len(batch)}"
        video, audio, v_label, a_label, o_label = batch
        
        print("\n--- BATCH SHAPES ---")
        print(f"Video: {video.shape}")
        print(f"Audio: {audio.shape}")
        print(f"Video Label: {v_label.shape}")
        print(f"Audio Label: {a_label.shape}")
        print(f"Overall Label: {o_label.shape}")
        
        assert video.shape == (2, 16, 3, 224, 224)
        assert audio.shape == (2, 3, 224, 224)
        assert v_label.shape == (2, 1)
        assert a_label.shape == (2, 1)
        assert o_label.shape == (2, 1)
        
        # Test Compatibility with V2 Model
        print("\n--- MODEL COMPATIBILITY TEST ---")
        model = MultiHeadDeepfakeModel(pretrained=False)
        out = model(video=video, audio=audio, return_all=True)
        
        print(f"Image Head Output Shape: {out['image'].shape}")
        print(f"Audio Head Output Shape: {out['audio'].shape}")
        print(f"Fusion Head Output Shape: {out['fusion'].shape}")
        
        # To train the Image Head, we expand the video label
        expanded_v_label = v_label.repeat_interleave(16, dim=0)
        print(f"Expanded Video Label for Image Head Loss: {expanded_v_label.shape}")
        
        assert out['image'].shape == expanded_v_label.shape, "Image head predictions and expanded labels shape mismatch!"
        assert out['audio'].shape == a_label.shape, "Audio head predictions and labels shape mismatch!"
        assert out['fusion'].shape == o_label.shape, "Fusion head predictions and labels shape mismatch!"
        
        print("\nAll V2 DataLoader tests passed successfully!")
        break

if __name__ == "__main__":
    main()
