import numpy as np
import sys

class Get_masks:


    def __init__(self, args, net):
        self.args = args
        self.net = net
        self.all_masks = []

