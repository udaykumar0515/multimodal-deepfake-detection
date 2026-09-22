import torch
from models import MultiHeadDeepfakeModel
from training import MultiTaskFocalLoss
import torch.optim as optim

def main():
    print("V2 Multi-Task Training Pipeline Sanity Test")
    
    # 1. Initialize Model and Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MultiHeadDeepfakeModel(pretrained=False).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    
    # 2. Mock Batch (B=2)
    B = 2
    video = torch.randn(B, 16, 3, 224, 224).to(device)
    audio = torch.randn(B, 3, 224, 224).to(device)
    
    v_label = torch.randint(0, 2, (B, 1)).float().to(device)
    a_label = torch.randint(0, 2, (B, 1)).float().to(device)
    o_label = torch.randint(0, 2, (B, 1)).float().to(device)
    
    print("\n--- Forward Pass ---")
    optimizer.zero_grad()
    preds = model(video=video, audio=audio, return_all=True)
    
    print(f"Image Preds: {preds['image'].shape}")
    print(f"Audio Preds: {preds['audio'].shape}")
    print(f"Fusion Preds: {preds['fusion'].shape}")
    
    print("\n--- Loss Calculation ---")
    losses = criterion(preds, v_label, a_label, o_label)
    
    print(f"Total Loss: {losses['total'].item():.4f}")
    print(f"Image Loss: {losses['image'].item():.4f}")
    print(f"Audio Loss: {losses['audio'].item():.4f}")
    print(f"Fusion Loss: {losses['fusion'].item():.4f}")
    
    print("\n--- Backward Pass ---")
    losses['total'].backward()
    
    # Check gradients
    has_grad_vid = False
    for name, p in model.visual_encoder.named_parameters():
        if p.grad is not None:
            has_grad_vid = True
            break
            
    has_grad_aud = False
    for name, p in model.audio_encoder.named_parameters():
        if p.grad is not None:
            has_grad_aud = True
            break
            
    has_grad_img_head = False
    for name, p in model.image_head.named_parameters():
        if p.grad is not None:
            has_grad_img_head = True
            break
            
    print(f"Visual Encoder gradients populated: {has_grad_vid}")
    print(f"Audio Encoder gradients populated: {has_grad_aud}")
    print(f"Image Head gradients populated: {has_grad_img_head}")
    
    assert has_grad_vid and has_grad_aud and has_grad_img_head, "Gradients missing for some branches!"
    
    print("\n--- Optimizer Step ---")
    optimizer.step()
    
    # Check for NaN parameters
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN detected in parameter {name}"
        
    print("Sanity Test Passed Successfully!")

if __name__ == "__main__":
    main()
