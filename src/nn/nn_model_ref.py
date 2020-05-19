import copy
import torch
from torch.utils.data import DataLoader
from src.nn.DataLoader import DataLoader_cipher_binary
from src.nn.ModelBaseline import ModelPaperBaseline
import time
from tqdm import tqdm
import os
import torch.nn as nn
from torch.autograd import Variable

from src.utils.utils import F1_Loss, BCE_bit_Loss


class NN_Model_Ref:

    def __init__(self, args, writer, device, rng, path_save_model, cipher, creator_data_binary, path_save_model_train):
        """

        :param args:
        :param writer:
        :param device:
        :param rng:
        :param path_save_model:
        """
        self.args = args
        self.writer = writer
        self.path_save_model_train = path_save_model_train
        self.device =device
        self.rng =rng
        self.cipher = cipher
        self.path_save_model =path_save_model
        self.net = self.choose_model()
        self.creator_data_binary = creator_data_binary
        self.create_data()


    def choose_model(self):
        if self.args.type_model=="baseline":
            return ModelPaperBaseline(self.args).to(self.device)

    def create_data(self):
        self.X_train_nn_binaire, self.Y_train_nn_binaire, self.c0l_train_nn, self.c0r_train_nn, self.c1l_train_nn, c1r_train_nn = self.creator_data_binary.make_data(
            self.args.nbre_sample_train);
        self.X_val_nn_binaire, self.Y_val_nn_binaire, self.c0l_val_nn, self.c0r_val_nn, self.c1l_val_nn, self.c1r_val_nn = self.creator_data_binary.make_data(
           self.args.nbre_sample_eval);
        #print(self.X_train_nn_binaire.shape, self.X_val_nn_binaire.shape)



    def train_from_scractch(self, name_input):
        self.t = Variable(torch.Tensor([0.5]))
        data_train = DataLoader_cipher_binary(self.X_train_nn_binaire, self.Y_train_nn_binaire, self.device)
        dataloader_train = DataLoader(data_train, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=self.args.num_workers)
        data_val = DataLoader_cipher_binary(self.X_val_nn_binaire, self.Y_val_nn_binaire, self.device)
        dataloader_val = DataLoader(data_val, batch_size=self.args.batch_size,
                                      shuffle=False, num_workers=self.args.num_workers)
        self.dataloaders = {'train': dataloader_train, 'val': dataloader_val}


        if self.args.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr_nn,
                                          weight_decay=self.args.weight_decay_nn)
        if self.args.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr_nn,
                                          weight_decay=self.args.weight_decay_nn)
        if self.args.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr_nn,
                                          momentum=self.args.momentum_nn)
        if self.args.loss_type == "BCE":
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        if self.args.loss_type == "MSE":
            self.criterion = nn.MSELoss().to(self.device)
        if self.args.loss_type == "SmoothL1Loss":
            self.criterion = nn.SmoothL1Loss().to(self.device)
        if self.args.loss_type == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.loss_type == "F1":
            self.criterion = F1_Loss().to(self.device)
        #if loss_type == "Mix_loss":
        #    self.criterion = BCE_bit_Loss(arg.lambda_loss_mse,arg.lambda_loss_f1, arg.lambda_loss_bit).to(self.device)
        if self.args.scheduler_type == "None":
            self.scheduler = None
        if self.args.scheduler_type == "CyclicLR":
            step_size_up = self.args.demicycle_1 * (self.args.nbre_sample_train // self.args.batch_size)
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, self.args.base_lr, self.args.max_lr, step_size_up, cycle_momentum=False) # exponential
        #if arg.scheduler_type == "OneCycleLR":
        #    from torch.optim.lr_scheduler import OneCycleLR
        #    scheduler = OneCycleLR(optimizer_conv, max_lr=max_lr, total_steps=step_size_up)
        self.train(name_input)

    def train(self, name_input):
        since = time.time()
        phrase = self.args.cipher + " round " +str(self.args.nombre_round_eval) +" inputs " + name_input +" size dataset "+ str(self.args.nbre_sample_train)
        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_loss = 100
        best_acc = 0.0
        n_batches = self.args.batch_size
        for epoch in range(self.args.num_epochs):
            print('-' * 10)
            print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, self.args.num_epochs, best_acc))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()
                if phase == 'val':
                    self.net.eval()
                running_loss = 0.0
                nbre_sample = 0
                TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(
                    1).long()
                for i, data in enumerate(self.dataloaders[phase], 0):
                    inputs, labels = data
                    self.optimizer.zero_grad()
                    # forward + backward + optimize
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.net(inputs.to(self.device))
                        loss = self.criterion(outputs.squeeze(1), labels.to(self.device))
                        desc = 'loss: %.4f; ' % (loss.item())
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                            self.optimizer.step()
                            if self.scheduler is not None:
                                self.scheduler.step()
                        preds = (outputs.squeeze(1) > self.t.to(self.device)).float() * 1
                        TP += (preds.eq(1) & labels.eq(1)).cpu().sum()
                        TN += (preds.eq(0) & labels.eq(0)).cpu().sum()
                        FN += (preds.eq(0) & labels.eq(1)).cpu().sum()
                        FP += (preds.eq(1) & labels.eq(0)).cpu().sum()
                        TOT = TP + TN + FN + FP
                        desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                            (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(),
                            TN.item() * 1.0 / TOT.item(), FN.item() * 1.0 / TOT.item(),
                            FP.item() * 1.0 / TOT.item())
                        running_loss += loss.item() * n_batches
                        nbre_sample += n_batches
                epoch_loss = running_loss / nbre_sample
                acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
                print('{} Acc: {:.4f}'.format(
                    phase, acc))
                #print(desc)
                print()
                self.writer.add_scalar(phase + ' Loss ' + phrase,
                                  epoch_loss,
                                  epoch)
                self.writer.add_scalar(phase + ' Acc ' + phrase,
                                  acc,
                                  epoch)
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.net.state_dict())
                    torch.save({'epoch': epoch + 1, 'acc': best_loss, 'state_dict': self.net.state_dict()},
                               os.path.join(self.path_save_model, str(best_loss) + '_bestloss.pth.tar'))
                if phase == 'val' and acc >= best_acc:
                    best_acc = acc
                    torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': self.net.state_dict()},
                               os.path.join(self.path_save_model, str(best_acc) + '_bestacc.pth.tar'))
            print()
        torch.save({'epoch': epoch + 1, 'acc': acc, 'state_dict': self.net.state_dict()},
                   os.path.join(self.path_save_model_train, 'Gohr_'+self.args.type_model+'_best_nbre_sampletrain_' + str(self.args.nbre_sample_train)+ '.pth.tar'))



        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('Best val Acc: {:4f}'.format(best_acc))
        print()
        # load best model weights
        self.net.load_state_dict(best_model_wts)

    def load_nn(self):
        self.net.load_state_dict(torch.load(
            os.path.join(self.path_save_model_train, 'Gohr_'+self.args.type_model+'_best_nbre_sampletrain_' + str(self.args.nbre_sample_train)+ '.pth.tar')), strict=False)
        self.net.to(self.device)



