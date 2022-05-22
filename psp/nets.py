import torch
import torch.nn as nn
import torch.nn.functional as F  
import math
from packages import FaceGenerator, ArcFace
from lib.blocks import get_blocks, bottleneck_IR_SE, GradualStyleBlock


class GradualStyleEncoder(nn.Module):
    def __init__(self, stylegan_size=1024):
        super(GradualStyleEncoder, self).__init__()
        blocks = get_blocks(50)
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
            )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR_SE(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride
                    ))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        p2 = self._upsample_add(c3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out

class PSPEncoder(nn.Module):
    def __init__(self):
        super(PSPEncoder, self).__init__()

        self.ArcFace = ArcFace()
        self.Encoder = GradualStyleEncoder(stylegan_size=1024)
        self.face_generator = FaceGenerator()
        self.avg_latent = self.face_generator.avg_latent

    def forward(self, x):
        w_fake = self.Encoder(x)
        I_recon = self.face_generator.gen_face(self.avg_latent + w_fake, size=256)
        return I_recon
    
    def get_id(self, image):
        return self.ArcFace.get_id(image)
