import torch
import torch.nn as nn
import torchvision.models as models


class VGGLoss(nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()
        # Load pre-trained VGG-16 network
        vgg = models.vgg16(pretrained=True).features

        # Slice the network into 4 blocks to capture texture/style at different depth levels
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])

        # Freeze weights (acts as a fixed feature extractor, no gradients computed)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, fake, target):
        # Map GAN image range [-1, 1] to [0, 1] before ImageNet normalization
        fake = (fake + 1) / 2.0
        target = (target + 1) / 2.0

        mean = (
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(fake.device)
        )
        std = (
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(fake.device)
        )

        fake = (fake - mean) / std
        target = (target - mean) / std

        # Extract features across layers and accumulate style loss
        h_fake = self.slice1(fake)
        h_target = self.slice1(target)
        loss = self._gram_loss(h_fake, h_target)

        h_fake = self.slice2(h_fake)
        h_target = self.slice2(h_target)
        loss += self._gram_loss(h_fake, h_target)

        h_fake = self.slice3(h_fake)
        h_target = self.slice3(h_target)
        loss += self._gram_loss(h_fake, h_target)

        h_fake = self.slice4(h_fake)
        h_target = self.slice4(h_target)
        loss += self._gram_loss(h_fake, h_target)

        return loss

    def _gram_loss(self, x, y):
        # Compute Gram matrix (measures texture/style correlations while ignoring spatial geometry)
        b, c, h, w = x.size()
        x_view = x.view(b, c, h * w)
        y_view = y.view(b, c, h * w)

        x_gram = torch.bmm(x_view, x_view.transpose(1, 2)) / (c * h * w)
        y_gram = torch.bmm(y_view, y_view.transpose(1, 2)) / (c * h * w)

        return torch.nn.functional.mse_loss(x_gram, y_gram)