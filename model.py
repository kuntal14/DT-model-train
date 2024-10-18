import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)  # Adjust in_channels as needed
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Fully connected layer
        conv_output_size = ((input_size - 2) // 2 - 1)  # After conv1 and pooling
        conv_output_size = (conv_output_size - 2) // 2  # After conv2 and pooling
        self.fc = nn.Linear(32 * conv_output_size, 1)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage
input_size = 401
model = CNN1D(input_size)
print(model)

# To train multiple models
num_models = 8
models = [CNN1D(input_size) for _ in range(num_models)]

