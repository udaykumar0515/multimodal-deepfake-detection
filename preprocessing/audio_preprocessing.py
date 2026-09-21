import os
import subprocess
import torch
import torchaudio  # type: ignore # pyright: ignore # pyrefly: ignore [missing-import]
import torch.nn.functional as F  # type: ignore # pyright: ignore # pyrefly: ignore [missing-import]

try:
    import imageio_ffmpeg  # type: ignore # pyright: ignore # pyrefly: ignore [missing-import]
    HAVE_FFMPEG = True
except ImportError:
    HAVE_FFMPEG = False


class AudioExtractor:
    """
    Extracts audio from video files using FFmpeg.
    Ensures the output is 16000 Hz, mono, PCM WAV format, maintaining original duration.
    """
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        if not HAVE_FFMPEG:
            raise RuntimeError("imageio-ffmpeg is required for AudioExtractor.")
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.sample_rate = sample_rate
        self.channels = channels

    def extract(self, video_path: str, output_wav_path: str) -> bool:
        """
        Extracts audio and saves it to output_wav_path.
        Returns True if successful, raises an exception if it fails.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

        cmd = [
            self.ffmpeg_exe,
            '-y',  # Overwrite output
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            output_wav_path
        ]

        try:
            # We redirect stdout and stderr to DEVNULL to avoid terminal spam
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            if os.path.exists(output_wav_path):
                os.remove(output_wav_path)
            raise RuntimeError(f"FFmpeg extraction failed for {video_path}") from e


class SpectrogramGenerator:
    """
    Generates a log-mel spectrogram from a WAV file suitable for EfficientNet.
    Outputs a [3, 224, 224] tensor.
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, target_size: int = 224):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_size = target_size
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512,
            f_min=20,
            f_max=8000
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def generate(self, wav_path: str) -> torch.Tensor:
        """
        Generates the [3, 224, 224] log-mel spectrogram tensor.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Wav file not found: {wav_path}")

        waveform, sr = torchaudio.load(wav_path)
        
        if sr != self.sample_rate:
            # Resample if necessary (though our extraction enforces 16k)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.shape[0] > 1:
            # Convert to mono just in case
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Generate Mel Spectrogram: shape -> [1, n_mels, time_frames]
        mel_spec = self.mel_transform(waveform)
        
        # Convert to Log-Mel (dB)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Resize to (target_size, target_size) -> e.g. (224, 224)
        # Interpolate expects [batch, channels, height, width]
        # Current shape is [channels(1), n_mels(128), time]
        log_mel_spec = log_mel_spec.unsqueeze(0) # [1, 1, 128, time]
        
        resized = F.interpolate(
            log_mel_spec, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        resized = resized.squeeze(0) # [1, 224, 224]
        
        # Duplicate to 3 channels for EfficientNet compatibility
        final_tensor = resized.repeat(3, 1, 1) # [3, 224, 224]
        
        return final_tensor
