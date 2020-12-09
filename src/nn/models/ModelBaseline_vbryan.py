import torch
import torch.nn as nn
from torch.nn import functional as F




class ModelPaperBaseline_vbryan(nn.Module):

    def __init__(self, args):
        super(ModelPaperBaseline_vbryan, self).__init__()
        self.args = args
        self.word_size = args.word_size
        self.conv0 = nn.Conv1d(in_channels=len(self.args.inputs_type), out_channels=args.out_channel0, kernel_size=1)
        self.BN0 = nn.BatchNorm1d(args.out_channel0, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = args.numLayers

        self.conv1a = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bnconv1a = nn.BatchNorm2d(32)
        self.bnconv2a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bnconv1b = nn.BatchNorm2d(32)
        self.bnconv2b = nn.BatchNorm2d(32)
        self.conv1c = nn.Conv1d(32, 32, 3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bnconv1c = nn.BatchNorm2d(32)
        self.bnconv2c = nn.BatchNorm2d(32)

        self.conv3a = nn.Conv1d(32, 64, 3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bnconv3a = nn.BatchNorm2d(64)
        self.bnconv4a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv1d(64, 64, 3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bnconv3b = nn.BatchNorm2d(64)
        self.bnconv4b = nn.BatchNorm2d(64)
        self.conv3c = nn.Conv1d(64, 64, 3, stride=1, padding=1)
        self.conv4c = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bnconv3c = nn.BatchNorm2d(64)
        self.bnconv4c = nn.BatchNorm2d(64)

        self.conv5a = nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.conv6a = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bnconv5a = nn.BatchNorm2d(128)
        self.bnconv6a = nn.BatchNorm2d(128)
        self.conv5b = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.conv6b = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bnconv5b = nn.BatchNorm2d(128)
        self.bnconv6b = nn.BatchNorm2d(128)
        self.conv5c = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.conv6c = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bnconv5c = nn.BatchNorm2d(128)
        self.bnconv6c = nn.BatchNorm2d(128)


        self.conv1sc = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.BN1sc = nn.BatchNorm1d(64, eps=0.01, momentum=0.99)
        self.conv2sc = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.BN2sc = nn.BatchNorm1d(128, eps=0.01, momentum=0.99)





        self.fc1 = nn.Linear(128 * args.word_size, args.hidden1)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(args.hidden1, args.hidden1)
        self.BN6 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(args.hidden1, 1)


    def forward(self, x):
        x = x.view(-1, len(self.args.inputs_type), self.word_size)
        self.x_input = x
        x = F.relu(self.BN0(self.conv0(x)))
        shortcut = x.clone()
        self.shorcut = shortcut[0]
        self.x_dico = {}


        x = F.relu(self.bnconv1a(self.conv1a(x)))
        x = F.relu(self.bnconv2a(self.conv2a(x)))
        x = x + shortcut
        x = F.relu(self.bnconv1b(self.conv1b(x)))
        x = F.relu(self.bnconv2b(self.conv2b(x)))
        x = x + shortcut
        x = F.relu(self.bnconv1c(self.conv1c(x)))
        x = F.relu(self.bnconv2c(self.conv2c(x)))
        x = x + shortcut

        shortcut = F.relu(self.BN1sc(self.conv1sc(shortcut)))

        x = F.relu(self.bnconv3a(self.conv3a(x)))
        x = F.relu(self.bnconv4a(self.conv4a(x)))
        x = x + shortcut
        x = F.relu(self.bnconv3b(self.conv3b(x)))
        x = F.relu(self.bnconv4b(self.conv4b(x)))
        x = x + shortcut
        x = F.relu(self.bnconv3c(self.conv3c(x)))
        x = F.relu(self.bnconv4c(self.conv4c(x)))
        x = x + shortcut

        shortcut = F.relu(self.BN2sc(self.conv2sc(shortcut)))

        x = F.relu(self.bnconv5a(self.conv5a(x)))
        x = F.relu(self.bnconv6a(self.conv6a(x)))
        x = x + shortcut
        x = F.relu(self.bnconv5b(self.conv5b(x)))
        x = F.relu(self.bnconv6b(self.conv6b(x)))
        x = x + shortcut
        x = F.relu(self.bnconv5c(self.conv5c(x)))
        x = F.relu(self.bnconv6c(self.conv6c(x)))
        x = x + shortcut

        x = x.view(x.size(0), -1)
        self.intermediare0 = x.clone()
        x = F.relu(self.BN5(self.fc1(x)))
        self.intermediare = x.clone()
        x = F.relu(self.BN6(self.fc2(x)))
        self.intermediare2 = x.clone()
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def freeze(self):
        self.conv0.weight.requires_grad = False
        self.BN0.bias.requires_grad = False
        for i in range(self.numLayers - 1):
            self.layers_conv[i].weight.requires_grad = False
            self.layers_batch[i].weight.requires_grad = False



