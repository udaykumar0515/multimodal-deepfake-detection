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
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup paths
project_root = Path(os.path.abspath(__file__)).parent
sys.path.append(str(project_root))

from models.multihead_model import MultiHeadDeepfakeModel
from preprocessing.video_preprocessing import VideoPreprocessor, RetinaFaceCropper, VisualTransform
from preprocessing.audio_preprocessing import AudioExtractor, SpectrogramGenerator

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detection V2 (Multi-Task)",
    layout="wide"
)

CHECKPOINT_PATH = project_root / "checkpoints" / "best_model.pt"

# ---------------------------------------------------------
# UTILITY CLASSES (GRAD-CAM)
# ---------------------------------------------------------
class GradCAM:
    """Reusable Grad-CAM implementation for V2."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, video=None, audio=None, image=None, target_head='image'):
        self.model.zero_grad()
        
        preds = self.model(video=video, audio=audio, image=image, return_all=True)
        target_logits = preds[target_head]
        
        prob = torch.sigmoid(target_logits).mean()
        prob.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activations, axis=1, keepdims=True)
        cam = np.maximum(cam, 0)
        
        cam_min = cam.min(axis=(2, 3), keepdims=True)
        cam_max = cam.max(axis=(2, 3), keepdims=True)
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return heatmap, prob.item()

def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def create_overlay(original_img, heatmap):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadDeepfakeModel(pretrained=False).to(device)
    
    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        st.stop()
        
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Enable gradients for Grad-CAM
    for param in model.parameters():
        param.requires_grad = True
        
    return model, device

@st.cache_resource
def load_preprocessors():
    video_prep = VideoPreprocessor(num_frames=16, margin=0.20, is_train=False)
    audio_ext = AudioExtractor()
    spec_gen = SpectrogramGenerator()
    face_cropper = RetinaFaceCropper(margin=0.20)
    vis_transform = VisualTransform(is_train=False)
    
    additional_targets = {f'image{i}': 'image' for i in range(1, 16)}
    multi_video_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets=additional_targets)
    
    return video_prep, audio_ext, spec_gen, face_cropper, vis_transform, multi_video_transform

# ---------------------------------------------------------
# INFERENCE LOGIC
# ---------------------------------------------------------
def process_video(video_path):
    model, device = load_model()
    video_prep, audio_ext, spec_gen, _, _, multi_video_transform = load_preprocessors()
    
    with st.spinner("Extracting and preprocessing video frames..."):
        raw_crops, failures = video_prep.process(video_path, return_raw_crops=True)
        if raw_crops is None:
            return None, f"Video preprocessing failed: {failures[0]['reason']}"
            
        video_np = np.stack(raw_crops).astype(np.uint8)
        kwargs = {'image': video_np[0]}
        for i in range(1, 16):
            kwargs[f'image{i}'] = video_np[i]
            
        transformed = multi_video_transform(**kwargs)
        frames = [transformed['image']]
        for i in range(1, 16):
            frames.append(transformed[f'image{i}'])
            
        video_tensor = torch.stack(frames).unsqueeze(0).to(device)
        video_tensor.requires_grad_(True)
        
    with st.spinner("Extracting and preprocessing audio..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            
        try:
            audio_ext.extract(video_path, temp_wav_path)
            spec_tensor = spec_gen.generate(temp_wav_path)
            audio_tensor = spec_tensor.unsqueeze(0).to(device)
            audio_tensor.requires_grad_(True)
        except Exception as e:
            os.remove(temp_wav_path)
            return None, f"Audio extraction failed: {e}"
        os.remove(temp_wav_path)
        
    with st.spinner("Running multimodal model & Grad-CAM..."):
        cam_video = GradCAM(model, model.visual_encoder.grad_cam_layer)
        cam_audio = GradCAM(model, model.audio_encoder.grad_cam_layer)
        
        # We need fusion probability
        with torch.no_grad():
            preds_fusion = model(video=video_tensor, audio=audio_tensor, return_all=True)
            fus_prob = torch.sigmoid(preds_fusion['fusion']).item()
            
        # Image Head Grad-CAM
        hm_video, vis_prob = cam_video.generate(video=video_tensor, audio=audio_tensor, target_head='image')
        # Audio Head Grad-CAM
        hm_audio, aud_prob = cam_audio.generate(video=video_tensor, audio=audio_tensor, target_head='audio')
        
    return {
        "vis_prob": vis_prob,
        "aud_prob": aud_prob,
        "fus_prob": fus_prob,
        "video_tensor": video_tensor.squeeze(0).cpu().detach().numpy(),
        "audio_tensor": audio_tensor.squeeze(0).cpu().detach().numpy(),
        "hm_video": hm_video[:, 0, :, :], # (16, H, W)
        "hm_audio": hm_audio[0, 0] # (H, W)
    }, None

def process_image(image_file):
    model, device = load_model()
    _, _, _, face_cropper, vis_transform, _ = load_preprocessors()
    
    with st.spinner("Preprocessing image..."):
        img = Image.open(image_file).convert("RGB")
        img_np = np.array(img)
        crop, info = face_cropper.crop_face(img_np)
        if crop is None:
            import cv2
            crop = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_CUBIC)
        tensor = vis_transform.apply(crop).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        
    with st.spinner("Running Image Analysis..."):
        cam_video = GradCAM(model, model.visual_encoder.grad_cam_layer)
        
        # We pass the image as 'image' parameter
        hm_video, vis_prob = cam_video.generate(image=tensor, target_head='image')
        
    return {
        "vis_prob": vis_prob,
        "image_tensor": tensor.squeeze(0).cpu().detach().numpy(),
        "hm_image": hm_video[0, 0, :, :]
    }, None

def process_audio(audio_file):
    model, device = load_model()
    _, _, spec_gen, _, _, _ = load_preprocessors()
    
    with st.spinner("Preprocessing audio..."):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio_file.read())
            temp_wav_path = temp_wav.name
            
        try:
            spec_tensor = spec_gen.generate(temp_wav_path)
            audio_tensor = spec_tensor.unsqueeze(0).to(device)
            audio_tensor.requires_grad_(True)
        except Exception as e:
            os.remove(temp_wav_path)
            return None, f"Audio processing failed: {e}"
        os.remove(temp_wav_path)
        
    with st.spinner("Running Audio Analysis..."):
        cam_audio = GradCAM(model, model.audio_encoder.grad_cam_layer)
        hm_audio, aud_prob = cam_audio.generate(audio=audio_tensor, target_head='audio')
        
    return {
        "aud_prob": aud_prob,
        "audio_tensor": audio_tensor.squeeze(0).cpu().detach().numpy(),
        "hm_audio": hm_audio[0, 0]
    }, None


# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "Results & Evaluation"])

def show_prediction_box(title, prob):
    is_fake = prob > 0.5
    label = "FAKE" if is_fake else "REAL"
    color = "#FF4B4B" if is_fake else "#00CC96"
    
    st.markdown(
        f"""
        <div style="border:2px solid {color}; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
            <h3 style="margin:0; color: {color}">{title}: {label}</h3>
            <p style="margin:0; font-size: 16px;">Fake Prob: {prob*100:.1f}%</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if page == "Detection":
    st.title("Deepfake Detection V2")
    st.write("Upload media to analyze using the independent V2 modalities.")
    
    modality = st.radio("Input Type:", ["Video", "Image", "Audio"], horizontal=True)
    
    if modality == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            if st.button("Analyze Video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_video_path = tfile.name
                
                results, error = process_video(temp_video_path)
                os.remove(temp_video_path)
                
                if error:
                    st.error(error)
                else:
                    st.subheader("Prediction Result")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        show_prediction_box("Visual / Video", results["vis_prob"])
                    with c2:
                        show_prediction_box("Audio", results["aud_prob"])
                    with c3:
                        show_prediction_box("Overall", results["fus_prob"])
                        
                    st.markdown("---")
                    st.subheader("Visual Grad-CAM (Image Head)")
                    
                    frames_to_show = [0, 7, 15]
                    vc1, vc2, vc3 = st.columns(3)
                    cols = [vc1, vc2, vc3]
                    
                    for idx, f_idx in enumerate(frames_to_show):
                        orig_img = denormalize_image(results["video_tensor"][f_idx])
                        overlay = create_overlay(orig_img, results["hm_video"][f_idx])
                        with cols[idx]:
                            st.image(orig_img, caption=f"Frame {f_idx} (Original)", width="stretch")
                            st.image(overlay, caption=f"Frame {f_idx} (Grad-CAM)", width="stretch")
                            
                    st.markdown("---")
                    st.subheader("Audio Grad-CAM (Audio Head)")
                    orig_aud = denormalize_image(results["audio_tensor"])
                    overlay_aud = create_overlay(orig_aud, results["hm_audio"])
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.image(orig_aud, caption="Spectrogram (Original)", width="stretch")
                    with ac2:
                        st.image(overlay_aud, caption="Spectrogram (Grad-CAM)", width="stretch")

    elif modality == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            if st.button("Analyze Image"):
                results, error = process_image(uploaded_file)
                if error:
                    st.error(error)
                else:
                    st.subheader("Image Analysis")
                    show_prediction_box("Image Prediction", results["vis_prob"])
                    
                    st.markdown("---")
                    st.subheader("Visual Grad-CAM (Image Head)")
                    orig_img = denormalize_image(results["image_tensor"])
                    overlay = create_overlay(orig_img, results["hm_image"])
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(orig_img, caption="Cropped Face (Original)", width="stretch")
                    with c2:
                        st.image(overlay, caption="Grad-CAM Overlay", width="stretch")

    elif modality == "Audio":
        uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            if st.button("Analyze Audio"):
                results, error = process_audio(uploaded_file)
                if error:
                    st.error(error)
                else:
                    st.subheader("Audio Analysis")
                    show_prediction_box("Audio Prediction", results["aud_prob"])
                    
                    st.markdown("---")
                    st.subheader("Audio Grad-CAM (Audio Head)")
                    orig_aud = denormalize_image(results["audio_tensor"])
                    overlay = create_overlay(orig_aud, results["hm_audio"])
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(orig_aud, caption="Spectrogram (Original)", width="stretch")
                    with c2:
                        st.image(overlay, caption="Grad-CAM Overlay", width="stretch")

elif page == "Results & Evaluation":
    st.title("V2 Results & Evaluation")
    
    metrics_path = project_root / "results" / "metrics" / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        st.write(f"**Test Sample Count:** {metrics['test_sample_count']} | **Best Checkpoint:** Epoch {metrics['checkpoint_epoch']}")
        
        st.subheader("Performance Metrics (Held-out Test Set)")
        c1, c2, c3 = st.columns(3)
        
        def render_metrics(col, title, data):
            with col:
                st.markdown(f"### {title}")
                for k, v in data.items():
                    st.markdown(f"**{k}:** {v*100:.2f}%" if k != "ROC-AUC" else f"**{k}:** {v:.4f}")
                    
        render_metrics(c1, "Image Head", metrics["image_metrics"])
        render_metrics(c2, "Audio Head", metrics["audio_metrics"])
        render_metrics(c3, "Fusion Head", metrics["fusion_metrics"])
        
    st.markdown("---")
    st.subheader("Four-Category Modality Diagnostic")
    st.write("Demonstrates V2's successful decoupling of independent prediction heads on the test set.")
    
    cat_path = project_root / "results" / "modality_analysis" / "modality_category_results.csv"
    if cat_path.exists():
        df = pd.read_csv(cat_path)
        st.dataframe(df, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Confusion Matrices")
    c1, c2, c3 = st.columns(3)
    c1.image(str(project_root / "results" / "confusion_matrices" / "image_confusion_matrix.png"), use_container_width=True)
    c2.image(str(project_root / "results" / "confusion_matrices" / "audio_confusion_matrix.png"), use_container_width=True)
    c3.image(str(project_root / "results" / "confusion_matrices" / "fusion_confusion_matrix.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("ROC Curves")
    c1, c2, c3 = st.columns(3)
    c1.image(str(project_root / "results" / "roc_curves" / "image_roc_curve.png"), use_container_width=True)
    c2.image(str(project_root / "results" / "roc_curves" / "audio_roc_curve.png"), use_container_width=True)
    c3.image(str(project_root / "results" / "roc_curves" / "fusion_roc_curve.png"), use_container_width=True)
