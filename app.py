import os
import sys
import tempfile
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup paths to ensure we can import our modules
project_root = Path(os.path.abspath(__file__)).parent
sys.path.append(str(project_root))

from models.fusion_model import MultimodalDeepfakeModel
from preprocessing import VideoPreprocessor
from preprocessing.audio_preprocessing import AudioExtractor, SpectrogramGenerator

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detection Using Multimodal Learning",
    layout="wide"
)

CHECKPOINT_PATH = project_root / "checkpoints" / "best_model.pt"

# ---------------------------------------------------------
# UTILITY CLASSES (GRAD-CAM)
# ---------------------------------------------------------
class GradCAM:
    """Reusable Grad-CAM implementation for a specific target layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self):
        """Generates heatmap from captured gradients and activations."""
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured.")
            
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam_min = cam.min(axis=(1,2), keepdims=True)
        cam_max = cam.max(axis=(1,2), keepdims=True)
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return heatmap

def denormalize_image(tensor):
    """Reverts ImageNet normalization for visualization (H, W, C)."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def create_overlay(original_img, heatmap):
    """Creates a colored heatmap overlay on the original RGB image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 0.5, heatmap_color, 0.5, 0)
    return overlay

# ---------------------------------------------------------
# CACHED FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the model and weights once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalDeepfakeModel(pretrained=False).to(device)
    
    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        st.stop()
        
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # We must enable gradients for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True
        
    return model, device

@st.cache_resource
def load_preprocessors():
    """Loads preprocessing tools."""
    video_prep = VideoPreprocessor(num_frames=16, margin=0.20, is_train=False)
    audio_ext = AudioExtractor()
    spec_gen = SpectrogramGenerator()
    
    additional_targets = {f'image{i}': 'image' for i in range(1, 16)}
    video_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets=additional_targets)
    
    return video_prep, audio_ext, spec_gen, video_transform

# ---------------------------------------------------------
# INFERENCE LOGIC
# ---------------------------------------------------------
def process_and_infer(video_path):
    model, device = load_model()
    video_prep, audio_ext, spec_gen, video_transform = load_preprocessors()
    
    with st.spinner("Extracting and preprocessing video frames..."):
        raw_crops, failures = video_prep.process(video_path, return_raw_crops=True)
        if raw_crops is None:
            st.error(f"Video preprocessing failed: {failures[0]['reason']}")
            return None
            
        video_np = np.stack(raw_crops).astype(np.uint8)
        
        kwargs = {'image': video_np[0]}
        for i in range(1, 16):
            kwargs[f'image{i}'] = video_np[i]
            
        transformed = video_transform(**kwargs)
        
        frames = [transformed['image']]
        for i in range(1, 16):
            frames.append(transformed[f'image{i}'])
            
        video_tensor = torch.stack(frames).unsqueeze(0).to(device) # (1, 16, 3, 224, 224)
        
    with st.spinner("Extracting and preprocessing audio..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            
        try:
            audio_ext.extract(video_path, temp_wav_path)
            spec_tensor = spec_gen.generate(temp_wav_path) # (3, 224, 224)
            audio_tensor = spec_tensor.unsqueeze(0).to(device)
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            os.remove(temp_wav_path)
            return None
            
        os.remove(temp_wav_path)
        
    with st.spinner("Running multimodal model..."):
        # Setup Grad-CAM
        cam_video = GradCAM(model, model.video_encoder.grad_cam_layer)
        cam_audio = GradCAM(model, model.audio_encoder.grad_cam_layer)
        
        model.zero_grad()
        logits = model(video_tensor, audio_tensor)
        prob_fake = torch.sigmoid(logits).item()
        
        logits.backward(retain_graph=True)
        
        hm_video = cam_video.generate_heatmap() # (16, 7, 7)
        hm_audio = cam_audio.generate_heatmap() # (1, 7, 7)
        
    return {
        "prob_fake": prob_fake,
        "video_tensor": video_tensor.squeeze(0).cpu().numpy(),
        "audio_tensor": audio_tensor.squeeze(0).cpu().numpy(),
        "hm_video": hm_video,
        "hm_audio": hm_audio[0]
    }

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "Results & Evaluation"])

if page == "Detection":
    st.title("Deepfake Detection Using Multimodal Learning")
    st.write("Upload a video to analyze it for deepfake manipulation using both video and audio modalities.")
    
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        if st.button("Analyze Video"):
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name
                
            try:
                result = process_and_infer(temp_video_path)
                
                if result:
                    prob = result['prob_fake']
                    prediction = "FAKE" if prob >= 0.5 else "REAL"
                    
                    st.subheader("Prediction Result")
                    
                    # Simple color for distinction without being overly flashy
                    color = "red" if prediction == "FAKE" else "green"
                    st.markdown(f"### Prediction: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
                    st.write(f"**Fake Probability:** {prob * 100:.2f}%")
                    
                    st.divider()
                    
                    st.subheader("Video Grad-CAM")
                    st.write("Visual attribution over selected temporal frames.")
                    
                    # Show 3 representative frames (start, middle, end)
                    target_frames = [0, 7, 15]
                    cols = st.columns(3)
                    
                    for idx, f_idx in enumerate(target_frames):
                        orig_frame = result["video_tensor"][f_idx]
                        hm = result["hm_video"][f_idx]
                        
                        orig_img = denormalize_image(orig_frame)
                        overlay_img = create_overlay(orig_img, hm)
                        
                        with cols[idx]:
                            st.write(f"**Frame {f_idx}**")
                            # Stack original and overlay vertically
                            st.image(orig_img, caption="Original", use_container_width=True)
                            st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)
                            
                    st.divider()
                    
                    st.subheader("Audio Grad-CAM")
                    st.write("Visual attribution over the Log-Mel Spectrogram.")
                    
                    orig_audio = result["audio_tensor"]
                    hm_a = result["hm_audio"]
                    
                    orig_audio_img = denormalize_image(orig_audio)
                    overlay_audio_img = create_overlay(orig_audio_img, hm_a)
                    
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.image(orig_audio_img, caption="Original Spectrogram", use_container_width=True)
                    with col_a2:
                        st.image(overlay_audio_img, caption="Grad-CAM Overlay", use_container_width=True)
                        
            finally:
                os.remove(temp_video_path)

elif page == "Results & Evaluation":
    st.title("Results & Evaluation")
    st.write("Final evaluation metrics on the held-out test dataset.")
    
    metrics_path = project_root / "results" / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        st.subheader("Test Set Metrics")
        
        # Display simple metric values
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        col2.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        col3.metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        col4.metric("F1-Score", f"{metrics.get('f1', 0)*100:.2f}%")
        col5.metric("ROC-AUC", f"{metrics.get('auc', 0)*100:.2f}%")
    else:
        st.info("Test metrics not found.")
        
    st.divider()
    
    st.subheader("Evaluation Visualizations")
    col_v1, col_v2 = st.columns(2)
    
    cm_path = project_root / "results" / "confusion_matrix.png"
    roc_path = project_root / "results" / "roc_curve.png"
    
    with col_v1:
        if cm_path.exists():
            st.image(Image.open(cm_path), caption="Confusion Matrix", use_container_width=True)
        else:
            st.write("Confusion matrix missing.")
            
    with col_v2:
        if roc_path.exists():
            st.image(Image.open(roc_path), caption="ROC Curve", use_container_width=True)
        else:
            st.write("ROC curve missing.")
