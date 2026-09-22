# Testing Data

This directory is **ONLY** for manual Streamlit demonstration and testing.

**CRITICAL RULES:**
- It is **NOT** part of the training dataset.
- Files placed here must **NOT** be used for training.
- Image folders represent image/visual labels.
- Audio folders represent audio labels.
- Video folders represent the four independent video/audio combinations used to verify the multi-task independent heads.

### Folder Mapping
*   `images/real/` - Authentic unmanipulated images
*   `images/fake/` - Synthetically manipulated images
*   `audio/real/` - Authentic pristine audio
*   `audio/fake/` - Synthesized/deepfake audio
*   `videos/real_video_real_audio/` - Both modalities authentic
*   `videos/fake_video_real_audio/` - Visually manipulated, audio authentic
*   `videos/real_video_fake_audio/` - Visually authentic, audio manipulated
*   `videos/fake_video_fake_audio/` - Both modalities manipulated
