import numpy as np

class BasicBlock:
    def __init__(self, in_channels, out_channels, projection=False):
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01

        self.projection = projection
        self.W_proj = (
            np.random.randn(in_channels, out_channels) * 0.01
            if projection
            else None
        )

    def forward(self, x):
        # Residual branch: Linear -> ReLU -> Linear
        out = np.dot(x, self.W1)
        out = np.maximum(0, out)
        out = np.dot(out, self.W2)

        # Shortcut branch
        if self.projection:
            shortcut = np.dot(x, self.W_proj)
        else:
            shortcut = x

        # Residual addition + ReLU
        out = out + shortcut
        out = np.maximum(0, out)

        return out


class ResNet18:
    def __init__(self, num_classes):
        # Initial "conv" layer: 3 -> 64
        self.conv1 = np.random.randn(3, 64) * 0.01

        # Stage 1: 64 -> 64 (2 blocks)
        self.layer1 = [
            BasicBlock(64, 64, projection=False),
            BasicBlock(64, 64, projection=False),
        ]

        # Stage 2: 64 -> 128 (first block uses projection)
        self.layer2 = [
            BasicBlock(64, 128, projection=True),
            BasicBlock(128, 128, projection=False),
        ]

        # Stage 3: 128 -> 256 (first block uses projection)
        self.layer3 = [
            BasicBlock(128, 256, projection=True),
            BasicBlock(256, 256, projection=False),
        ]

        # Stage 4: 256 -> 512 (first block uses projection)
        self.layer4 = [
            BasicBlock(256, 512, projection=True),
            BasicBlock(512, 512, projection=False),
        ]

        # Final fully connected layer: 512 -> num_classes
        self.fc = np.random.randn(512, num_classes) * 0.01

    def forward(self, x):
        # Initial conv + ReLU
        x = np.dot(x, self.conv1)
        x = np.maximum(0, x)

        # Residual stages
        for block in self.layer1:
            x = block.forward(x)

        for block in self.layer2:
            x = block.forward(x)

        for block in self.layer3:
            x = block.forward(x)

        for block in self.layer4:
            x = block.forward(x)

        # Classification logits
        logits = np.dot(x, self.fc)
        return logits


def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    # Initial layer
    out = np.maximum(0, np.dot(x, conv1))

    # Block 1 (identity shortcut)
    residual = out
    out = np.maximum(0, np.dot(out, W1_b1))
    out = np.dot(out, W2_b1)
    out = np.maximum(0, out + residual)

    # Block 2 (projection shortcut)
    residual = np.dot(out, Ws_b2)
    out = np.maximum(0, np.dot(out, W1_b2))
    out = np.dot(out, W2_b2)
    out = np.maximum(0, out + residual)

    # Final classifier
    logits = np.dot(out, fc)
    return logits