# Deepfake Detection Using Multimodal Learning — Implementation Blueprint

## 1. Project Overview
**Title:** Deepfake Detection Using Multimodal Learning
**Goal:** Build a robust, production-ready multimodal deepfake detection system capable of processing and classifying fake and real inputs across three modalities:
- Images
- Audio
- Videos (Multimodal)

The system is augmented with Explainable AI (XAI) using Grad-CAM to visually explain its decisions. 
**Hardware Constraint:** The entire pipeline is explicitly designed to train and run inference efficiently on an RTX 4050 Laptop GPU (6GB VRAM).

## 2. Dataset Analysis
**Dataset:** FakeAVCeleb
FakeAVCeleb is used as the sole dataset for this project. It is a comprehensive dataset containing both video and audio, making it inherently suitable for multimodal training. An initial audit has already been completed, and potential identity leakages have been analyzed and mitigated.

**Visual Branch Label Distribution (Video Modality):**
*   Total Real: 1,000
*   Total Fake: 20,566
*   Ratio (Real:Fake): 1:20.57
*   Train: Real=700, Fake=14,509
*   Validation: Real=150, Fake=3,019
*   Test: Real=150, Fake=3,038

**Audio Branch Label Distribution (Audio Modality):**
*   Total Real: 10,209
*   Total Fake: 11,357
*   Ratio (Real:Fake): 1:1.11
*   Train: Real=7,178, Fake=8,031
*   Validation: Real=1,497, Fake=1,672
*   Test: Real=1,534, Fake=1,654

**Video Duration & Frame Stats:**
*   Average duration: 5.48 seconds
*   Min/Max duration: 2.04 sec / 17.68 sec
*   Average FPS: 25.04
*   Average frame count: ~137 frames

## 3. Dataset Split
**Identity-Based Split**
The dataset is split into training (70%), validation (15%), and testing (15%) sets based strictly on subject identity (`source`). This ensures that a person appearing in the training set never appears in the validation or test sets. This identity isolation entirely prevents the model from memorizing specific faces or voices, forcing it to learn generalized deepfake artifacts instead.

## 4. Video Preprocessing
**Face Detection: RetinaFace**
RetinaFace handles face detection throughout the pipeline. 
*   **Why it was selected:** RetinaFace provides superior robustness, especially for side profiles, varied lighting conditions, and low-resolution frames, significantly outperforming legacy models like MTCNN. 
*   **Implementation Details:** The algorithm detects the face bounding box and applies a 20% margin to ensure the blending boundary (where fake faces are stitched onto real heads) is captured. The cropped facial region is then resized to 224x224 pixels, the native input size for the visual backbone, normalized using ImageNet statistics.

## 5. Frame Sampling
**Uniform Temporal Sampling to 16 Frames**
For video inputs, the output tensor size is ALWAYS fixed to 16 frames. The algorithm does NOT simply extract the first 16 frames of the video. Instead, it samples frames uniformly across the entire length of the video.
*   **Examples:**
    *   If a video contains 48 frames, the algorithm will sample approximately every 3 frames (e.g., frames 0, 3, 6, 9...).
    *   If a video contains 320 frames, it will sample approximately every 20 frames (e.g., frames 0, 20, 40, 60...).
*   **Why this is better:** The spacing automatically adapts to the video length while always producing exactly 16 frames. This guarantees temporal coverage across the entire video, keeps input tensor sizes perfectly fixed for efficient GPU batching, avoids oversampling short videos (which causes duplicate frames), and avoids undersampling long videos (which misses the end of the video). This is a standard strategy in modern video recognition models.

## 6. Visual Data Augmentation
**Conservative Spatial Augmentations**
Augmentations are carefully chosen to prevent the destruction of subtle high-frequency artifacts left by deepfake generators (e.g., blending lines, compression artifacts).
*   **Selected Parameters:**
    *   Horizontal Flip: probability = 0.5
    *   Rotation: ±10° (faces naturally do not tilt excessively)
    *   Color Jitter: Brightness=0.1, Contrast=0.1, Saturation=0.1, Hue=0.05
    *   Random Erasing: probability = 0.1, scale=(0.02, 0.1) (Simulates occlusion like hands over the face)
*   **Why aggressive augmentations are avoided:** Techniques like heavy Gaussian blur, aggressive JPEG compression, or large cutouts will effectively mask or destroy the very blending boundaries and frequency inconsistencies the CNN needs to detect.

## 7. Audio Preprocessing
**Audio Representation: Log Mel Spectrogram**
Audio is converted into a 2D image-like representation before being processed by the CNN.
*   **Audio Extraction:** Audio is extracted from the video file using FFmpeg.
*   **16 kHz & Mono:** The audio is resampled to 16,000 Hz and converted to a single mono channel. This standardizes all inputs and sufficiently captures human voice frequencies and synthesis artifacts.
*   **Mel Spectrogram:** The waveform is transformed into a Mel Spectrogram (128 mel bins).
*   **Log Scaling:** A logarithmic scale is applied to expand the dynamic range, making quiet synthesis artifacts mathematically prominent.
*   **Resize & 3-Channel Conversion:** The 2D spectrogram is resized to 224x224 pixels using bicubic interpolation. Because the visual backbone expects RGB images, the single-channel spectrogram is duplicated across three channels (or mapped via a colormap).
*   **Input to EfficientNet:** The resulting 3-channel 224x224 "image" is then normalized and fed directly into the audio backbone.

## 8. Visual Model
**Visual Backbone: EfficientNet-B0**
EfficientNet-B0 serves as the visual feature extractor. 
*   **Why it was selected:** It offers an optimal balance between accuracy and computational efficiency, fitting perfectly within the 6GB VRAM constraint while delivering state-of-the-art CNN performance.
*   **Transfer Learning:** The model is initialized with ImageNet pre-trained weights. ImageNet features (edges, textures, gradients) are highly effective starting points for finding visual inconsistencies in deepfakes.
*   **Image Inference:** A single static image passes through the network to generate a 1280-d feature vector.
*   **Video Inference:** The fixed tensor of 16 frames is passed through the network, generating 16 separate 1280-d feature vectors.
*   **Grad-CAM Compatibility:** As a pure Convolutional Neural Network, EfficientNet-B0 natively supports Grad-CAM on its final convolutional block without requiring complex custom hooks or adapters.

## 9. Audio Model
**Audio Backbone: EfficientNet-B0**
The audio backbone uses the exact same architecture as the visual backbone.
*   **Why it was selected:** By converting audio into Log Mel Spectrograms, audio classification becomes an image classification problem. ImageNet pre-trained weights transfer exceptionally well to spectrogram textures.
*   **Simplifying the Pipeline:** Using the identical EfficientNet-B0 backbone for both modalities greatly simplifies the project implementation, maintenance, and debugging. The codebase only needs to define and manage one CNN architecture. The fusion process is also simpler as both branches output vectors of the exact same size and latent space characteristics.

## 10. Feature Extraction & Temporal Aggregation
The pipeline extracts dense feature vectors from the raw inputs.
*   **Image:** A single image is processed into a single 1280-dimensional feature vector.
*   **Video:** 16 frames are processed into 16 separate feature vectors (shape: `[16, 1280]`). 
*   **Mean Pooling (Temporal Aggregation):** To reduce the temporal dimension, the 16 frame vectors are averaged together. Mathematically, this is the mean along the sequence dimension: `Output = 1/16 * sum(frame_features)`. 
    *   **Why Mean Pooling:** It provides a stable, unified 1280-dimensional feature vector representing the entire video. It was selected over Max Pooling (which is prone to noise spikes) and Attention Pooling (which adds unnecessary parameters and complexity) because it is simple, parameter-free, and acts as a strong, reliable baseline.
*   **Audio:** The spectrogram is processed into a single 1280-dimensional feature vector.

## 11. Fusion Strategy
**Feature Concatenation**
For multimodal video inputs, visual and audio features must be combined.
*   **Implementation:** The 1280-dimensional visual feature vector is concatenated directly with the 1280-dimensional audio feature vector.
*   **Result:** A single 2560-dimensional multimodal feature vector.
*   **Why it was selected:** Feature concatenation preserves 100% of the information from both modalities. It is highly reliable, easily debuggable, and passes all information directly to the MLP classifier, allowing the linear layers to dynamically learn which modality's features are more important.

## 12. Classifier
**Multi-Layer Perceptron (MLP)**
The final classification is performed by an MLP head.
*   **Architecture:** 
    1.  Linear Layer (Input: 2560, Output: 512)
    2.  ReLU Activation
    3.  Dropout (p=0.3)
    4.  Linear Layer (Input: 512, Output: 1)
    5.  Sigmoid Activation (Applied during inference or implicitly via `BCEWithLogitsLoss`).
*   **Implementation Details:** For single-modality inputs (Image-only or Audio-only), parallel MLPs are instantiated with input sizes of 1280. The dropout layer is crucial for preventing the model from overfitting to the relatively small dataset.

## 13. Training Strategy
**Transfer Learning, Focal Loss, and AdamW**
The entire pipeline is trained end-to-end.
*   **Complete Training Pipeline:** The EfficientNet backbones (loaded with ImageNet weights) and the randomly initialized MLP classifier are trained together. Gradients flow from the MLP all the way back to the CNN layers, ensuring the feature extractors adapt specifically to deepfake detection.
*   **Focal Loss:** The dataset analysis revealed a massive 1:20 class imbalance in the visual branch. Standard Binary Cross Entropy (BCE) would fail completely by simply predicting "Fake" every time. Focal Loss is the finalized decision because it dynamically scales the loss based on prediction confidence, down-weighting the abundant "Fake" examples (via alpha) and forcing the model to focus on the hard, deceptive "Real" examples (via gamma).
*   **AdamW:** Selected because it correctly implements weight decay for Adam, providing better generalization and preventing the model weights from growing too large during fine-tuning.

## 14. Model Outputs
This section clearly defines the inputs and outputs for every phase of inference.

*   **Visual Model (Standalone Image/Video)**
    *   **Input:** Face Crop (3x224x224)
    *   **Output:** 1280-d Feature Vector, Single Probability Score (0.0 = Real, 1.0 = Fake)
*   **Audio Model (Standalone Audio)**
    *   **Input:** Log Mel Spectrogram (3x224x224)
    *   **Output:** 1280-d Feature Vector, Single Probability Score (0.0 = Real, 1.0 = Fake)
*   **Fusion Model (Multimodal Video)**
    *   **Input:** 2560-d Concatenated Feature Vector
    *   **Output:** Final Probability Score (0.0 = Real, 1.0 = Fake)
*   **Explainability**
    *   **Input:** Target CNN Layer Activations + Class Gradients
    *   **Output:** 2D Heatmap (overlaying the original input image/spectrogram) highlighting areas of high suspicion.

## 15. Explainability (Grad-CAM)
**Gradient-weighted Class Activation Mapping**
Grad-CAM provides visual explanations for the model's decisions, vastly increasing the trustworthiness of the system.
*   **How it works:** It uses the gradient of the target class (Fake) flowing into the final convolutional layer of the EfficientNet-B0 backbone to produce a coarse localization map highlighting important regions in the image.
*   **No Retraining:** Because Grad-CAM is a post-hoc analysis tool that relies entirely on backpropagating gradients through an already-frozen model during inference, it requires absolutely zero retraining or architectural modifications.
*   **Image Heatmaps:** Overlays a heatmap directly onto the facial crop, pointing out visual artifacts like blurry teeth, mismatched skin tones, or blending lines.
*   **Video Frame Heatmaps:** Applies the exact same process to key frames extracted from the video sequence.
*   **Audio Spectrogram Heatmaps:** Overlays a heatmap on the Log Mel Spectrogram, highlighting the specific time and frequency bands containing vocal synthesis artifacts.
*   **Streamlit Presentation:** The final web application will display the uploaded media side-by-side with the generated Grad-CAM heatmaps, providing an intuitive, immediately understandable explanation to the user.

## 16. Deployment
**Framework: Streamlit**
The final application will be developed using Streamlit. Streamlit allows for the rapid development of clean, interactive, Python-native web interfaces. The application will feature upload components for images, audio files, and videos. Upon upload, it will run the preprocessing pipeline, execute inference, and display the final probability score alongside the Grad-CAM visualizations in a modern, user-friendly layout.

## 17. Project Directory Structure
The repository will be structured to ensure maintainability and separation of concerns:
```text
Deepfake_Detection/
├── data/              # Stores the raw FakeAVCeleb dataset and split CSVs
├── preprocessing/     # Scripts for RetinaFace cropping, FFmpeg extraction, and Log Mel Specs
├── models/            # PyTorch architecture definitions (EfficientNet, MLP, Fusion)
├── training/          # Training loop, loss functions (Focal Loss), optimizers
├── evaluation/        # Scripts for calculating AUC-ROC, Accuracy, F1, and generating confusion matrices
├── explainability/    # Grad-CAM implementation and visualization scripts
├── app/               # Streamlit application files (app.py)
└── configs/           # YAML or JSON files storing hyperparameters (batch size, learning rate)
```

## 18. Implementation Roadmap
The step-by-step roadmap for software development:
1.  **Repository setup** (Directory structure, requirements.txt, environment)
2.  **Dataset CSV generation** (Creating the Identity-based train/val/test splits)
3.  **Video preprocessing** (Implementing RetinaFace and Uniform Temporal Sampling)
4.  **Audio preprocessing** (Implementing audio extraction and Log Mel Spectrograms)
5.  **Dataset classes** (Writing PyTorch `Dataset` and `DataLoader` logic)
6.  **Visual model** (Instantiating EfficientNet-B0 and the visual MLP head)
7.  **Audio model** (Instantiating EfficientNet-B0 and the audio MLP head)
8.  **Fusion model** (Implementing feature concatenation and the multimodal MLP)
9.  **Training** (Writing the training loop, Focal Loss, and checkpoint saving)
10. **Evaluation** (Testing on the isolated set, generating metrics)
11. **Grad-CAM** (Implementing post-hoc explainability)
12. **Streamlit** (Building the final user interface)

---

## 19. Current Project Status

### FINALIZED
*   Dataset & Label Distributions
*   Dataset Split Strategy (Identity-based)
*   Face Detection (RetinaFace)
*   Frame Sampling (Uniform Temporal Sampling to 16 Frames)
*   Visual Data Augmentation (Conservative parameters)
*   Audio Representation (Log Mel Spectrogram)
*   Visual Backbone (EfficientNet-B0)
*   Audio Backbone (EfficientNet-B0)
*   Temporal Aggregation (Mean Pooling)
*   Fusion Strategy (Feature Concatenation)
*   Classifier Architecture (MLP)
*   Training Approach (Transfer Learning)
*   Loss Function (Focal Loss)
*   Optimizer (AdamW)
*   Explainability (Grad-CAM)
*   Evaluation Metrics
*   Deployment Framework (Streamlit)

### NOT STARTED
*   *Coding Phase 1:* Data Preprocessing and PyTorch Loaders
*   *Coding Phase 2:* Model Architectures and Training Scripts
*   *Coding Phase 3:* Evaluation and Grad-CAM Implementation
*   *Coding Phase 4:* Streamlit Application Development
