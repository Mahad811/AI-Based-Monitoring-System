"""
Fall Classifier Module (V2 - Temporal RGB Triplets)

Loads EfficientNet-B0 trained on 3-frame temporal RGB and performs inference.
Supports both single model and 5-fold ensemble.
"""

import torch
import torch.nn as nn
import numpy as np
import timm
from pathlib import Path


class FallClassifier:
    """
    Binary classifier (fall vs normal) using EfficientNet-B0.
    Input: temporal RGB images (224x224) using RGB stacking:
        - R = grayscale(frame[t-1])  # Past frame
        - G = grayscale(frame[t])     # Current frame (appearance)
        - B = grayscale(frame[t+1])   # Future frame
    Includes appearance information in G channel for temporal context.
    Supports 5-model ensemble for improved robustness.
    """
    
    def __init__(self, model_path, device='auto', use_ensemble=True):
        """
        Args:
            model_path: Path to trained model weights (best.pt or directory with fold*.pt)
            device: 'auto', 'cuda', or 'cpu'
            use_ensemble: If True and fold models exist, use ensemble
        """
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load models
        model_path = Path(model_path)
        self.models = []
        
        # Load ensemble models (fold0.pt ... fold4.pt)
        if use_ensemble and model_path.is_dir():
            fold_paths = sorted(model_path.glob('fold*.pt'))
            if len(fold_paths) >= 3:
                print(f"Loading {len(fold_paths)} fall classifier models (ensemble)...")
                for fold_path in fold_paths:
                    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2,
                                             drop_rate=0.5, drop_path_rate=0.2)
                    model.load_state_dict(torch.load(fold_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                print(f"✓ Loaded {len(self.models)} fall classifier models")
            else:
                # Fallback to single best.pt in directory
                best_path = model_path / 'best.pt'
                if best_path.exists():
                    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
                    model.load_state_dict(torch.load(best_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                    print(f"✓ Loaded single fall classifier model (best.pt)")
        else:
            # Single model file
            if model_path.is_file():
                model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"✓ Loaded single fall classifier model")
            else:
                raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # ImageNet normalization (used during training)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        # Class names (assumes ImageFolder order: fall=0, normal=1)
        self.class_names = ['fall', 'normal']
    
    def preprocess(self, temporal_rgb):
        """
        Preprocess temporal RGB image for inference
        
        Args:
            temporal_rgb: (224, 224, 3) numpy array [0-255]
            
        Returns:
            tensor: (1, 3, 224, 224) normalized tensor
        """
        # Convert to float and normalize to [0, 1]
        img = temporal_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Normalize with ImageNet stats
        img = (img - self.mean) / self.std
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
    
    def classify(self, temporal_rgb):
        """
        Classify a temporal RGB image (with ensemble if available)
        
        Args:
            temporal_rgb: (224, 224, 3) numpy array from TemporalEncoder
            
        Returns:
            dict with keys:
                - 'class': 'fall' or 'normal'
                - 'confidence': probability of predicted class (0-1)
                - 'fall_prob': probability of fall class (0-1)
                - 'normal_prob': probability of normal class (0-1)
        """
        if temporal_rgb is None:
            return None
        
        # Preprocess
        img_tensor = self.preprocess(temporal_rgb).to(self.device)
        
        # Inference with ensemble
        with torch.no_grad():
            if len(self.models) == 1:
                # Single model
                output = self.models[0](img_tensor)
                probs = torch.softmax(output, dim=1)
            else:
                # Ensemble: average probabilities from all models
                all_probs = []
                for model in self.models:
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    all_probs.append(probs)
                
                # Average across all models
                probs = torch.stack(all_probs).mean(dim=0)
            
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        return {
            'class': self.class_names[pred_idx],
            'confidence': confidence,
            'fall_prob': probs[0, 0].item(),
            'normal_prob': probs[0, 1].item()
        }
    
    def classify_batch(self, temporal_rgb_batch):
        """
        Classify multiple temporal RGB images
        
        Args:
            temporal_rgb_batch: list of (224, 224, 3) numpy arrays
            
        Returns:
            list of classification dicts
        """
        return [self.classify(img) for img in temporal_rgb_batch]
