import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import math

#Bi-Real-net

class ModelPaperBaseline_bin2(nn.Module):

    def __init__(self, args):
        super(ModelPaperBaseline_bin2, self).__init__()
        self.args = args
        #Conv1d = conv1d_Q_fn(w_bit=1)
        Linear = linear_Q_fn(w_bit=1)
        self.act_q = activation_quantize_fn(a_bit=2)
        self.word_size = args.word_size
        self.binary_activation = BinaryActivation()
        self.conv0 = HardBinaryConv(len(self.args.inputs_type), args.out_channel0, kernel_size=1)
        self.BN0 = nn.BatchNorm1d(args.out_channel0, eps=0.01, momentum=0.99)
        self.layers_conv = nn.ModuleList()
        self.layers_batch = nn.ModuleList()
        self.numLayers = args.numLayers
        for i in range(args.numLayers - 1):
            self.layers_conv.append(HardBinaryConv(args.out_channel1, args.out_channel1, kernel_size=3, padding=1))
            self.layers_batch.append(nn.BatchNorm1d(args.out_channel1, eps=0.01, momentum=0.99))
        self.fc1 = Linear(args.out_channel1 * args.word_size, args.hidden1)  # 6*6 from image dimension
        self.BN5 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc2 = Linear(args.hidden1, args.hidden1)
        self.BN6 = nn.BatchNorm1d(args.hidden1, eps=0.01, momentum=0.99)
        self.fc3 = Linear(args.hidden1, 1)



    def forward(self, x):
        x = x.view(-1, len(self.args.inputs_type), self.word_size)
        self.x_input = x
        x = self.binary_activation(self.BN0(self.conv0(x)))
        shortcut = x.clone()
        self.shorcut = shortcut[0]
        for i in range(len(self.layers_conv)):
            x = self.layers_conv[i](x)
            x = self.layers_batch[i](x)
            x = self.binary_activation(x)
            x = x + shortcut
        x = x.view(x.size(0), -1)
        x = self.act_q(F.relu(self.BN5(self.fc1(x))))
        self.intermediare = x.clone()
        x = self.act_q(F.relu(self.BN6(self.fc2(x))))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def freeze(self):
        self.conv0.weight.requires_grad = False
        self.BN0.bias.requires_grad = False
        for i in range(self.numLayers - 1):
            self.layers_conv[i].weight.requires_grad = False
            self.layers_batch[i].weight.requires_grad = False



def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * self.uniform_q(weight) - 1)
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=0):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)


    def forward(self, x):
        real_weight = self.weight
        scaling_factor = torch.mean(torch.mean(abs(real_weight),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weight)
        cliped_weights = torch.clamp(real_weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        return out

def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q


