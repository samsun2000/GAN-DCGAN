import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


#TODO!! Part 2. Finish the following two definition of classes to define Generator and Discriminitor model structure
# Define the Generator Network

# Generator model
class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.nz = params['nz']
        self.nc = params['nc']
        self.n_classes = params['n_classes']

        # Embedding for categorical labels
        self.embedding = nn.Embedding(self.n_classes, self.nz)

        # Modify the input of the first layer to accept the concatenated noise and label embeddings
        self.deconv1 = nn.ConvTranspose2d(self.nz * 2, 256, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, self.nc, 4, 2, 1, bias=False)

    def forward(self, z, labels):
        # Convert labels to embeddings
        labels = self.embedding(labels)
        # Reshape input and labels and concatenate them
        z = z.view(z.size(0), self.nz, 1, 1)
        labels = labels.unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, labels], 1)  # concatenate noise and label embeddings
        x = F.relu(self.bn1(self.deconv1(z)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.nc = params['nc']
        self.n_classes = params['n_classes']

        # Embedding for categorical labels
        self.embedding = nn.Embedding(self.n_classes, 32*32)

        # Modify the input of the first layer to accept the concatenated image and label embeddings
        self.conv1 = nn.Conv2d(self.nc + 1, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        # Convert labels to embeddings
        labels = self.embedding(labels).view(x.size(0), 1, 32, 32)
        # Concatenate images and label embeddings
        x = torch.cat([x, labels], 1)  # concatenate image and label embeddings
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.sigmoid(self.conv4(x))
        return x

#This is normal DCGAN version for single class character
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def weights_init(w):
#     """
#     Initializes the weights of the layer, w.
#     """
#     classname = w.__class__.__name__
#     if classname.find('conv') != -1:
#         nn.init.normal_(w.weight.data, 0.0, 0.02)
#     elif classname.find('bn') != -1:
#         nn.init.normal_(w.weight.data, 1.0, 0.02)
#         nn.init.constant_(w.bias.data, 0)
#
#
# #TODO!! Part 2. Finish the following two definition of classes to define Generator and Discriminitor model structure
# # Define the Generator Network
# class Generator(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.nz = params['nz']
#         # Input is the latent vector Z.
#         # Based on the structure from the image, we'll define the deconv layers:
#
#         # 1*1*100 -> 4*4*256
#         self.deconv1 = nn.ConvTranspose2d(self.nz, 256, 4, 1, 0, bias=False)
#         self.bn1 = nn.BatchNorm2d(256)
#
#         # 4*4*256 -> 8*8*128
#         self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#
#         # 8*8*128 -> 16*16*64
#         self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(64)
#
#         # 16*16*64 -> 32*32*1
#         self.deconv4 = nn.ConvTranspose2d(64, params['nc'], 4, 2, 1, bias=False)
#
#     def forward(self, x):
#         # Reshape the input tensor to 1*1*100
#         x = x.view(x.size(0), self.nz, 1, 1)
#
#         # Defining how the data flows in Generator
#         x = F.relu(self.bn1(self.deconv1(x)))
#         x = F.relu(self.bn2(self.deconv2(x)))
#         x = F.relu(self.bn3(self.deconv3(x)))
#         x = torch.tanh(self.deconv4(x))
#
#         return x
#
#
# class Discriminator(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.nc = params['nc']
#         # Input Dimension: (nc) x 32 x 32
#         self.conv1 = nn.Conv2d(self.nc, 64, kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#
#         self.conv4 = nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=0, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.sigmoid(self.conv4(x))
#         return x