from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange






class DataLoader_curriculum(Dataset):
    """"""

    def __init__(self, X, Y, device, old_net, catgeorie, train = True):
        """
        """
        self.categorie_1 = []
        self.categorie_2 = []
        self.categorie_3 = []
        self.old_net = old_net
        self.X, self.Y = X, Y
        self.device = device
        self.train = train
        self.catgeorie = catgeorie
        self.t = Variable(torch.Tensor([0.5]))
        if self.train:
            print("START PREPROCESSING")
            self.oder_input()
        self.categorie_22 = self.categorie_1 + self.categorie_2


    def __len__(self):
        if self.train:
            if self.catgeorie == 1:
                return len(self.categorie_1)
            if self.catgeorie == 2:
                return len(self.categorie_22)
            if self.catgeorie == 3:
                return len(self.Y)
        else:
            return len(self.Y)


    def __getitem__(self, idx):
        if self.train:
            if self.catgeorie == 1:
                input_x, input_y = self.X[self.categorie_1[idx]], self.Y[self.categorie_1[idx]]
                return torch.tensor(input_x).float(), torch.tensor(input_y).float()
            if self.catgeorie == 2:
                input_x, input_y = self.X[self.categorie_22[idx]], self.Y[self.categorie_22[idx]]
                return torch.tensor(input_x).float(), torch.tensor(input_y).float()
            if self.catgeorie == 3:
                input_x, input_y = self.X[idx], self.Y[idx]
                return torch.tensor(input_x).float(), torch.tensor(input_y).float()
        else:
            input_x, input_y = self.X[idx], self.Y[idx]
            return torch.tensor(input_x).float(), torch.tensor(input_y).float()



    def oder_input(self):
        self.old_net.eval()
        with torch.no_grad():
            for index in trange(len(self.X)):
                inputs, labels = torch.tensor(self.X[index]).float().to(self.device), torch.tensor(self.Y[index]).float().to(self.device)
                outputs = self.old_net(inputs.unsqueeze(0))
                predicted = (outputs.squeeze(1) > self.t.to(self.device)).float() * 1
                if predicted == labels:
                    if self.Y[index] ==1:
                        if outputs > 0.8:
                            self.categorie_1.append(index)
                        if 0.8 > outputs > 0.6:
                            self.categorie_2.append(index)
                        if 0.7 > outputs > 0.:
                            self.categorie_3.append(index)
                    if self.Y[index] == 0:
                        if outputs < 0.2:
                            self.categorie_1.append(index)
                        if 0.4 > outputs > 0.2:
                            self.categorie_2.append(index)
                        if 1.0 > outputs > 0.4:
                            self.categorie_3.append(index)
        print("Nbre input categorie_1:", len(self.categorie_1))
        print("Nbre input categorie_2:", len(self.categorie_2))
        print("Nbre input categorie_3:", len(self.categorie_3))
        print("Nbre input all:", len(self.categorie_1)+len(self.categorie_2)+len(self.categorie_3))




#data_train = DataLoader_speck()
#print(data_train.X.shape, data_train.Y.shape)
