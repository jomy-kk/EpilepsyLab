# 2. Setup the autoencoder

import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Zero padding is almost the same as average padding in this case
        # Input = b, 1, 4, 300
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (4,15), stride=1, padding=(0,7)), # b, 8, 1, 300
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2), # b, 8, 1, 150
            nn.Conv2d(8, 4, 3, stride=1, padding=1), # b, 8, 1, 150
            nn.Tanh(),
            nn.MaxPool2d((1,2), stride=2) # b, 4, 1, 75
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=(0,1)), # b, 8, 1, 150
            nn.Tanh(),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=(0,1), output_padding=1), # b, 8, 4, 300
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1), # b, 1, 4, 300
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def setup_model():
    return ConvAutoEncoder()
