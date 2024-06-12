from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential, Module


class LeNet5(Module):
    """
    From https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
    """
    def __init__(self, input_channels, padding_conv1, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = Conv2d(input_channels, 6, 5, 1, padding_conv1)
        self.bn1 = BatchNorm2d(6)
        self.relu1conv = ReLU()
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.bn2 = BatchNorm2d(16)
        self.relu2conv = ReLU()
        self.pool2 = MaxPool2d(2, 2)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.relu1fc = ReLU()
        self.fc2 = Linear(120, 84)
        self.relu2fc = ReLU()
        self.fc3 = Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1conv(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2conv(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu1fc(x)
        x = self.fc2(x)
        x = self.relu2fc(x)
        x = self.fc3(x)
        return x


class LeNet5_MNIST(LeNet5):
    def __init__(self, num_classes):
        super(LeNet5_MNIST, self).__init__(input_channels=1, padding_conv1='same', num_classes=num_classes)


class LeNet5_CIFAR(LeNet5):
    def __init__(self, num_classes):
        super(LeNet5_CIFAR, self).__init__(input_channels=3, padding_conv1=0, num_classes=num_classes)
