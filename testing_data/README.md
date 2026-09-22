# Testing Data

This directory contains manually selected **FakeAVCeleb samples** extracted from the main dataset.

### Purpose
*   They are used exclusively for **Streamlit demonstration and manual testing**.
*   They are **NOT** training data.
*   They are **NOT** added to any train/validation/test manifests.

### Directory Structure

#### Image Samples
*   `images/real/`: Authentic unmanipulated images (frames extracted from real videos)
*   `images/fake/`: Synthetically manipulated images (frames extracted from fake videos)

#### Audio Samples
*   `audio/real/`: Authentic pristine audio 
*   `audio/fake/`: Synthesized/deepfake audio

#### Video Categories
The four independent video/audio combinations used to verify the multi-task independent heads:
*   `videos/real_video_real_audio/` - Both modalities authentic
*   `videos/fake_video_real_audio/` - Visually manipulated, audio authentic
*   `videos/real_video_fake_audio/` - Visually authentic, audio manipulated
*   `videos/fake_video_fake_audio/` - Both modalities manipulated
