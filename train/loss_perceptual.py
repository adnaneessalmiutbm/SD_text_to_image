# loss_perceptual.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


class VGGLoss(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()

        vgg = models.vgg16(
            weights=VGG16_Weights.IMAGENET1K_FEATURES
        ).features

        self.blocks = nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ])

        for b in self.blocks:
            b.to(device=device, dtype=torch.float32)
            for p in b.parameters():
                p.requires_grad = False

    def forward(self, x_hat, x):
        x_hat = x_hat.float()
        x = x.float()

        mean = torch.tensor([0.485, 0.456, 0.406], device=x_hat.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x_hat.device).view(1, 3, 1, 1)

        x_hat = (x_hat - mean) / std
        x     = (x     - mean) / std

        loss = 0.0

        with torch.cuda.amp.autocast(enabled=False):
            for block in self.blocks:
                x_hat = block(x_hat)
                with torch.no_grad():
                    x = block(x)
                loss = loss + F.mse_loss(x_hat, x)

        return loss
