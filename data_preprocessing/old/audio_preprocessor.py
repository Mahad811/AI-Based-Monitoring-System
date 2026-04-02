"""
Audio Preprocessing Pipeline
Processes raw audio data for distress detection and keyword spotting
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json


class AudioPreprocessor:
    """Preprocess audio for distress detection and keyword spotting"""
    
    def __init__(self, config):
        """
        Initialize audio preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_sr = 16000  # 16kHz sample rate
        self.target_duration = 2.0  # 2 seconds
        
    def process_audio(self, audio_path, output_dir):
        """
        Process a single audio file
        
        Args:
            audio_path: Path to input audio
            output_dir: Directory to save processed audio
            
        Returns:
            dict: Processing results
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Normalize amplitude
            audio = librosa.util.normalize(audio)
            
            # Remove silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Pad or trim to target duration
            target_length = int(self.target_sr * self.target_duration)
            
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                # Trim
                audio = audio[:target_length]
            
            # Save processed audio
            audio_name = Path(audio_path).stem
            output_path = os.path.join(output_dir, f"{audio_name}_processed.wav")
            sf.write(output_path, audio, self.target_sr)
            
            return {
                'success': True,
                'input_path': audio_path,
                'output_path': output_path,
                'duration': len(audio) / self.target_sr,
                'sample_rate': self.target_sr
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'input_path': audio_path
            }
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for processed audio
            
        Returns:
            list: Processing results for all files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        results = []
        
        for audio_file in Path(input_dir).rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                print(f"Processing: {audio_file.name}")
                result = self.process_audio(str(audio_file), output_dir)
                results.append(result)
        
        return results
    
    def extract_features(self, audio_path):
        """
        Extract MFCC features for model training
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            numpy array: MFCC features
        """
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=32,
            n_fft=512,
            hop_length=160
        )
        
        # Normalize
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        
        return mfccs
    
    def create_train_val_test_split(self, dataset_dir, output_file,
                                    train_ratio=0.7, val_ratio=0.15):
        """
        Create train/val/test splits for audio dataset
        
        Args:
            dataset_dir: Directory containing processed audio
            output_file: Path to save split information
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            
        Returns:
            dict: Split statistics
        """
        audio_files = []
        for audio_file in Path(dataset_dir).glob('*.wav'):
            audio_files.append(str(audio_file))
        
        # Shuffle
        np.random.shuffle(audio_files)
        
        # Calculate split indices
        total = len(audio_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split
        train_files = audio_files[:train_end]
        val_files = audio_files[train_end:val_end]
        test_files = audio_files[val_end:]
        
        # Save splits
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        with open(output_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        return {
            'total': total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
    
    def augment_audio(self, audio_path, output_dir, augmentations=None):
        """
        Apply data augmentation to audio
        
        Args:
            audio_path: Input audio path
            output_dir: Output directory
            augmentations: List of augmentations
                         ['noise', 'shift', 'pitch', 'speed']
        """
        if augmentations is None:
            augmentations = ['noise', 'shift']
        
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        audio_name = Path(audio_path).stem
        
        os.makedirs(output_dir, exist_ok=True)
        
        for aug in augmentations:
            augmented = audio.copy()
            
            if aug == 'noise':
                # Add white noise
                noise = np.random.normal(0, 0.005, audio.shape)
                augmented = audio + noise
                
            elif aug == 'shift':
                # Time shift
                shift = np.random.randint(-sr//2, sr//2)
                augmented = np.roll(audio, shift)
                
            elif aug == 'pitch':
                # Pitch shift
                augmented = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                
            elif aug == 'speed':
                # Speed change
                augmented = librosa.effects.time_stretch(audio, rate=1.1)
            
            # Save augmented audio
            output_path = os.path.join(output_dir, f"{audio_name}_{aug}.wav")
            sf.write(output_path, augmented, sr)
    
    def create_dataset_from_long_audio(self, audio_path, output_dir, 
                                       segment_duration=2.0, overlap=0.5):
        """
        Split long audio file into segments
        
        Args:
            audio_path: Path to long audio file
            output_dir: Output directory for segments
            segment_duration: Duration of each segment (seconds)
            overlap: Overlap between segments (seconds)
        """
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        audio_name = Path(audio_path).stem
        
        os.makedirs(output_dir, exist_ok=True)
        
        segment_length = int(segment_duration * sr)
        hop_length = int((segment_duration - overlap) * sr)
        
        segment_idx = 0
        for start in range(0, len(audio) - segment_length, hop_length):
            segment = audio[start:start + segment_length]
            
            output_path = os.path.join(
                output_dir, 
                f"{audio_name}_seg_{segment_idx:03d}.wav"
            )
            sf.write(output_path, segment, sr)
            segment_idx += 1
        
        return segment_idx

