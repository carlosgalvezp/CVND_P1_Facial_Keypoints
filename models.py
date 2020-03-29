## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # Based on this paper:
        # https://arxiv.org/pdf/1710.00977.pdf       
        # This network is not fully convolutional, so it only works for inputs of size 1x224x224        
        # Changes w.r.t. the paper:
        # - Add one more conv layer to further reduce the final image size, to avoid a lot of parameters in Dense layers.
        # - Use filters of odd size (5x5, 3x3, 1x1).
        # - Remove "activation_6", since it's "linear" in the paper and doesn't add any extra value.
        self.conv1 = nn.Conv2d(in_channels=1,   out_channels=16,  kernel_size=5)  # Output after pool: (16, 110, 110)
        self.conv2 = nn.Conv2d(in_channels=16,  out_channels=32,  kernel_size=5)  # Output after pool: (32, 53, 53)
        self.conv3 = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=3)  # Output after pool: (64, 25, 25)
        self.conv4 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3)  # Output after pool: (128, 11, 11)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # Output after pool: (256, 5, 5)
        
        self.activation1_5 = nn.ELU()
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.dropout3 = nn.Dropout2d(p=0.3)
        self.dropout4 = nn.Dropout2d(p=0.4)
        self.dropout5 = nn.Dropout2d(p=0.5)
        self.dropout6 = nn.Dropout2d(p=0.6)
        
        # The input to the first FC layer will be of size (256, 5, 5) = 6400
        # The final output is a 136-dimensional vector, to predict 68 2-dimensional keypoints
        self.dense1 = nn.Linear(in_features=6400, out_features=1000)
        self.dense2 = nn.Linear(in_features=1000, out_features=1000)
        self.dense3 = nn.Linear(in_features=1000, out_features=136)        

        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # Convolutional layers
        x =               self.pool(self.activation1_5(self.conv1(x)))
        x = self.dropout1(self.pool(self.activation1_5(self.conv2(x))))
        x = self.dropout2(self.pool(self.activation1_5(self.conv3(x))))
        x = self.dropout3(self.pool(self.activation1_5(self.conv4(x))))
        x = self.dropout4(self.pool(self.activation1_5(self.conv5(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
                          
        # Fully-connected layers
        x = self.dropout5(self.activation1_5(self.dense1(x)))
        x = self.dropout6(                   self.dense2(x))
        x =                                  self.dense3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
