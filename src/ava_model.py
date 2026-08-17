import torch
import torch.nn as nn
import torch.nn.functional as F


"""
TinyTemporalCNN: Lightweight 1D-CNN for Edge Deployment

This model serves as our temporal architecture for seizure prediction. 
It is specifically designed to solve two major project constraints:

1. Microcontroller Deployment: By using 1D convolutions instead of LSTMs 
   or Transformers, the model learns sequential time-series patterns while 
   keeping the parameter count and SRAM usage exceptionally low. This ensures 
   it can easily be compiled into a C++ header for edge hardware later.

2. Dynamic Training Windows: It incorporates an AdaptiveAvgPool1d layer 
   right before the final classifier. This dynamically squashes the time 
   dimension, meaning the model will not crash if fed different sequence 
   lengths. This allows the team (e.g., Ethan) to freely experiment with 
   different training window sizes without needing to rewrite the architecture.

Architecture Flow:
- Conv1D + MaxPool (Blocks 1 & 2): Slides across the time sequence to extract 
  changing features and compresses the signal to save memory.
- Adaptive Pooling: Standardizes the output size regardless of input length.
- Linear Layer: Outputs the final binary prediction logits (Seizure vs. Interictal).
"""


class TinyTemporalCNN(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        """
        A highly efficient 1D-CNN designed specifically for microcontroller deployment.
        in_channels: The number of features or EEG channels being fed in per time step.
        """
        super(TinyTemporalCNN, self).__init__()
        
        # Temporal Block 1: Slide across the timeline
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Temporal Block 2: Extract deeper sequential patterns
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Adaptive pooling squashes the timeline down to a single value per channel
        # This makes the model resilient to different window lengths!
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        """
        Expected input shape: (batch_size, in_channels, sequence_length)
        """
        # Pass through temporal blocks
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Pool across the remaining time dimension and flatten
        x = self.global_pool(x)
        x = torch.flatten(x, 1) 
        
        # Output probability logits
        x = self.fc(x)
        return x


