import torch
import torch.nn as nn
from torch.nn import functional as F

from lambda_networks import LambdaLayer




class ModelPaperBaselineN_batch_lambda(nn.Module):

    def __init__(self, args):
        super(ModelPaperBaselineN_batch_lambda, self).__init__()
        self.args = args
        self.word_size = args.word_size
        self.layer0 = LambdaLayer(
            dim=len(self.args.inputs_type),  # channels going in
            dim_out=args.out_channel0,  # channels out
            n=args.word_size * args.Nbatch,  # number of input pixels (64 x 64 image)
            dim_k=16,  # key dimension
            heads=4,  # number of heads, for multi-query
            dim_u=1  # 'intra-depth' dimension
        )
        self.BN0 = nn.BatchNorm2d(args.out_channel0, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = args.numLayers
        for i in range(args.numLayers - 1):
            self.layers_conv.append(LambdaLayer(
            dim=args.out_channel1,  # channels going in
            dim_out=args.out_channel1,  # channels out
            n=args.word_size * args.Nbatch,  # number of input pixels (64 x 64 image)
            dim_k=16,  # key dimension
            heads=4,  # number of heads, for multi-query
            dim_u=1  # 'intra-depth' dimension
        ))
            self.layers_batch.append(nn.BatchNorm2d(args.out_channel1, eps=0.01, momentum=0.99))
        self.fc1 = nn.Linear(args.out_channel1 * args.word_size * args.Nbatch, args.hidden1)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc2 = nn.Linear(args.hidden1, args.hidden1)
        self.BN6 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc3 = nn.Linear(args.hidden1, 1)


    def forward(self, x):
        self.x_input = x
        x = F.relu(self.BN0(self.layer0(x)))
        shortcut = x.clone()
        self.shorcut = shortcut[0]
        self.x_dico = {}
        for i in range(len(self.layers_conv)):
            x = self.layers_conv[i](x)
            x = self.layers_batch[i](x)
            x = F.relu(x)
            x = x + shortcut
            self.x_dico[i] = x
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.BN5(self.fc1(x)))
        self.intermediare = x.clone()
        x = F.relu(self.BN6(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def freeze(self):
        self.conv0.weight.requires_grad = False
        self.BN0.bias.requires_grad = False
        for i in range(self.numLayers - 1):
            self.layers_conv[i].weight.requires_grad = False
            self.layers_batch[i].weight.requires_grad = False



