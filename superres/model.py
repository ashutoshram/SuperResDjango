import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.clamp_0_1 = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.leaky_relu = nn.LeakyReLU()
        """
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        """
        """
        self.conv1 = nn.Conv2d(1, 4, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(4, 4, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(4, 4, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(4, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        """
        """
        self.conv1 = nn.Conv2d(1, 2, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(2, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        """
        """
        self.conv1 = nn.Conv2d(1, 2, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(2, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        """
        """
        self.conv1 = nn.Conv2d(1, 2, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(2, 2, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(2, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        """
        self.conv1 = nn.Conv2d(1, 4, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(4, 4, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(4, 4, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(4, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.clamp_0_1(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))

        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        """

        """
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        """
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pixel_shuffle(self.relu(self.conv5(x)))
        """
        return x

    def forward_with_intermediates(self, x):
        layer1_x = self.relu(self.conv1(x))
        layer2_x = self.clamp_0_1(self.conv2(layer1_x))
        layer3_x = self.relu(self.conv3(layer2_x))
        layer4_x = self.pixel_shuffle(self.conv4(layer3_x))
        return(layer4_x, (layer1_x, layer2_x, layer3_x))

    def _initialize_weights(self):

        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

        """
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight, init.calculate_gain('relu'))
        """
