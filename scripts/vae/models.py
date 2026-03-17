import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class DecoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        
        # 1. PixelShuffle Upsampler for the Main Branch
        if scale > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * (scale**2), kernel_size=1, bias=False),
                nn.PixelShuffle(scale),
                nn.BatchNorm2d(in_channels)
            )
        else:
            self.upsample = nn.Identity()

        # 2. Convolutions (Processing the upsampled features)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 3. Shortcut Connection (Must match spatial and channel dimensions)
        self.shortcut = nn.Sequential()
        if scale != 1 or in_channels != out_channels:
            shortcut_layers = []
            if scale > 1:
                # Use PixelShuffle in the shortcut too for consistency
                shortcut_layers.append(nn.Conv2d(in_channels, in_channels * (scale**2), kernel_size=1, bias=False))
                shortcut_layers.append(nn.PixelShuffle(scale))
            
            shortcut_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            shortcut_layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        out = self.upsample(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)




################################################
################################################
################################################
################################################
################################################


################################################
################################################






class VAE(nn.Module):
    def __init__(self, latent_dim, blocks_per_layer):
        super().__init__()

        input_size = 128
        self.final_liyer_size = 4#input_size // 2**5
        
        # --- ENCODER ---
        self.encoder_input = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.en_layer1 = self._make_encoder_layer(64, 128, blocks_per_layer, stride=2)  # 64 -> 32
        self.en_layer2 = self._make_encoder_layer(128, 256, blocks_per_layer, stride=2) # 32 -> 16
        self.en_layer3 = self._make_encoder_layer(256, 512, blocks_per_layer, stride=2) # 16 -> 8
        self.en_layer4 = self._make_encoder_layer(512, 1024, blocks_per_layer, stride=2) # 8 -> 4
        
        self.fc_mu = nn.Linear(1024 * self.final_liyer_size * self.final_liyer_size, latent_dim)
        self.fc_logvar = nn.Linear(1024 * self.final_liyer_size * self.final_liyer_size, latent_dim)

        # --- DECODER ---
        self.fc_decode = nn.Linear(latent_dim, 1024 * self.final_liyer_size * self.final_liyer_size)
        
        self.de_layer4 = self._make_decoder_layer(1024, 512, blocks_per_layer, scale=2) # 4 -> 8
        self.de_layer3 = self._make_decoder_layer(512, 256, blocks_per_layer, scale=2) # 8 -> 16
        self.de_layer2 = self._make_decoder_layer(256, 128, blocks_per_layer, scale=2) # 16 -> 32
        self.de_layer1 = self._make_decoder_layer(128, 64, blocks_per_layer, scale=2)  # 32 -> 64

        self.decoder_output = nn.Sequential(
            DecoderResBlock(64, 64, scale=2), # 32 -> 64
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid() # Good practice if pixels are normalized [0, 1]
        )

    def _make_encoder_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _make_decoder_layer(self, in_channels, out_channels, blocks, scale):
        layers = [DecoderResBlock(in_channels, out_channels, scale=scale)]
        for _ in range(1, blocks):
            layers.append(DecoderResBlock(out_channels, out_channels, scale=1))
        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.encoder_input(x)
        x = self.en_layer1(x)
        x = self.en_layer2(x)
        x = self.en_layer3(x)
        x = self.en_layer4(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 1024, self.final_liyer_size, self.final_liyer_size)
        x = self.de_layer4(x)
        x = self.de_layer3(x)
        x = self.de_layer2(x)
        x = self.de_layer1(x)
        return self.decoder_output(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
# def vae_loss(recon_x, x, mu, logvar):
#     recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
#     kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     # kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#     return recon_loss + kld_loss, recon_loss, kld_loss

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # 1. Recon loss: sum over pixels/channels, mean over batch
    
    bce = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none')
    recon_loss = bce.view(x.size(0), -1).sum()#.sum(dim=1).mean()
    # recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')


    # 2. KLD loss: sum over latent dims, mean over batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld_loss = kld.mean()

    # 3. Combine with Beta weighting
    return recon_loss + (beta * kld_loss), recon_loss, kld_loss

# def vae_loss(recon_x, x, mu, logvar):
#     recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kl_loss