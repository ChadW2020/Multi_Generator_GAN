
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import time
import sys
import argparse

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.datasets as datasets

from models import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=int, required=True)
parser.add_argument('--contin', type=int, default=0)
args = parser.parse_args()

if (args.method == 0):
    CURRENT_WORK = "save_3G_mes/"
elif (args.method == 1):
    CURRENT_WORK = "save_3G_cwi/"
elif (args.method == 2):
    CURRENT_WORK = "save_3G_pmc/"
else:
    print("Wrong Input")
    sys.exit()

if not os.path.exists(CURRENT_WORK):
    os.makedirs(CURRENT_WORK)


def keep_larger_than(a, x):
    y = [n for n in a if n > x]
    if not y:
      return False
    return y

if (args.contin == 0):
    savelist = list(range(0, 11)) + list(range(15, 201, 5))
    num_epochs = list(range(0, 201))
else:
    num_epochs = list(range((args.contin + 1), 201))
    savelist = keep_larger_than(list(range(0, 11)) + list(range(15, 201, 5)), args.contin)
    

BATCH_SIZE = 128
INPUT_DIM = 1000
LR = 0.0002
G_STEPS = 1
D_STEPS = 10
LAMBDA = 10
K = 1

CUDA = "cuda"
device = torch.device(CUDA if torch.cuda.is_available() else "cpu")


def allocate_3_mes(y_fake_0, y_fake_1, y_fake_2):
    ### 1st method, madatory half-half (one third each in 3 generator case, of course).
    sorted_y0, indi_y0 = torch.sort(torch.squeeze(y_fake_0), descending=True)
    sorted_y1, indi_y1 = torch.sort(torch.squeeze(y_fake_1), descending=True)
    sorted_y2, indi_y2 = torch.sort(torch.squeeze(y_fake_2), descending=True)
    sorted_y = torch.stack((sorted_y0, sorted_y1, sorted_y2), dim=0)# [3, BATCH_SIZE]
    m_gen = sorted_y.size()[0]
    n_data = sorted_y.size()[1]
    quota = int(n_data / m_gen)
    winner_id = torch.transpose(torch.argsort(sorted_y, dim=0, descending=True), 0, 1).tolist()# Nested list, BATCH_SIZE sub lists, each length 3.
    unfilled = [0, 1, 2]
    indices = [[] for x in range(3)] 
    count = [0, 0, 0]
    for i in range(n_data):
        win_i = winner_id[i][0]
        indices[win_i].append(i)
        count[win_i] = count[win_i] + 1
        if (count[win_i] == quota):
            # Remove win_i from the rest sub lists
            unfilled.remove(win_i)
            for j in range(i, n_data):
                winner_id[j].remove(win_i)
        if (len(unfilled) == 1):
            # Rest samples all goes to the last generator left unfilled.
            indices[unfilled[0]] = indices[unfilled[0]] + list(range(i, n_data))
            break
        # Exception Throughout
        if ((len(unfilled) <= 0) or (max(count) > quota) or (count[win_i] > quota)):
            print("This is impossible. Check your code.")
            print("len(unfilled)")
            print(len(unfilled))
            print("max(count) - quota")
            print(max(count) - quota)
            print("count[win_i] - quota")
            print(count[win_i] - quota)
    
    choice_0 = indi_y0[indices[0]]
    choice_1 = indi_y1[indices[1]]
    choice_2 = indi_y2[indices[2]]
    return choice_0, choice_1, choice_2



def allocate_3_cwi(y_fake_0, y_fake_1, y_fake_2):
    ### Use 2nd method
    ### Compare with each generator itself
    # y_fake_0 = y_fake_0 - torch.mean(y_fake_0)
    # y_fake_1 = y_fake_1 - torch.mean(y_fake_1)
    # y_fake_2 = y_fake_2 - torch.mean(y_fake_2)
    y_fake_0 = y_fake_0 - torch.median(y_fake_0)
    y_fake_1 = y_fake_1 - torch.median(y_fake_1)
    y_fake_2 = y_fake_2 - torch.median(y_fake_2)
    
    sorted_y0, indi_y0 = torch.sort(torch.squeeze(y_fake_0), descending=True)
    sorted_y1, indi_y1 = torch.sort(torch.squeeze(y_fake_1), descending=True)
    sorted_y2, indi_y2 = torch.sort(torch.squeeze(y_fake_2), descending=True)
    sorted_y = torch.stack((sorted_y0, sorted_y1, sorted_y2), dim=0)# [3, BATCH_SIZE]
    winner = torch.argmax(sorted_y, dim=0)# [BATCH_SIZE]
    choice_0 = indi_y0[(winner == 0)]
    choice_1 = indi_y1[(winner == 1)]
    choice_2 = indi_y2[(winner == 2)]
    return choice_0, choice_1, choice_2


def allocate_3_pmc(y_fake_0, y_fake_1, y_fake_2):
    ### Use 3nd method
    ### Pure market competition
    sorted_y0, indi_y0 = torch.sort(torch.squeeze(y_fake_0), descending=True)
    sorted_y1, indi_y1 = torch.sort(torch.squeeze(y_fake_1), descending=True)
    sorted_y2, indi_y2 = torch.sort(torch.squeeze(y_fake_2), descending=True)
    sorted_y = torch.stack((sorted_y0, sorted_y1, sorted_y2), dim=0)# [3, BATCH_SIZE]
    winner = torch.argmax(sorted_y, dim=0)# [BATCH_SIZE]
    choice_0 = indi_y0[(winner == 0)]
    choice_1 = indi_y1[(winner == 1)]
    choice_2 = indi_y2[(winner == 2)]
    return choice_0, choice_1, choice_2



def train(G_0, G_1, G_2, D, G0_optim, G1_optim, G2_optim, D_optim, loader):
    mean_yf0 = []
    mean_yf1 = []
    mean_yf2 = []
    std_yf0 = []
    std_yf1 = []
    std_yf2 = []
    for batch_idx, (x_real, target) in enumerate(loader):
        print(x_real.size())
        x_real = x_real.to(device)
        for i in range(G_STEPS):
            G0_optim.zero_grad()
            G1_optim.zero_grad()
            G2_optim.zero_grad()
            D_optim.zero_grad()
            z_noise = torch.normal(0, 0.02, size=(x_real.size()[0], INPUT_DIM)).to(device)
            x_fake_0 = G_0(z_noise)
            x_fake_1 = G_1(z_noise)
            x_fake_2 = G_2(z_noise)
            y_fake_0 = D(x_fake_0)
            y_fake_1 = D(x_fake_1)
            y_fake_2 = D(x_fake_2)
            ### Record mean and std of y_fake for further observation
            mean_yf0.append(torch.mean(y_fake_0).item())
            mean_yf1.append(torch.mean(y_fake_1).item())
            mean_yf2.append(torch.mean(y_fake_2).item())
            std_yf0.append(torch.std(y_fake_0).item())
            std_yf1.append(torch.std(y_fake_1).item())
            std_yf2.append(torch.std(y_fake_2).item())
            
            if (args.method == 0):
                ### 1. Mandatory even shares
                choice_0, choice_1, choice_2 = allocate_3_mes(y_fake_0, y_fake_1, y_fake_2)
            elif (args.method == 1):
                ### 2. Compare the best of each set
                choice_0, choice_1, choice_2 = allocate_3_cwi(y_fake_0, y_fake_1, y_fake_2)
            elif (args.method == 2):
                ### 3. Pure market competition
                choice_0, choice_1, choice_2 = allocate_3_pmc(y_fake_0, y_fake_1, y_fake_2)
            
            x_fake_0 = x_fake_0[choice_0, :, :, :]
            x_fake_1 = x_fake_1[choice_1, :, :, :]
            x_fake_2 = x_fake_2[choice_2, :, :, :]
            y_fake_0 = y_fake_0[choice_0, :]
            y_fake_1 = y_fake_1[choice_1, :]
            y_fake_2 = y_fake_2[choice_2, :]
            loss_G0 = -torch.mean(y_fake_0)
            loss_G0.backward()
            loss_G1 = -torch.mean(y_fake_1)
            loss_G1.backward()
            loss_G2 = -torch.mean(y_fake_2)
            loss_G2.backward()
            G0_optim.step()
            G1_optim.step()
            G2_optim.step()
        for i in range(D_STEPS):
            G0_optim.zero_grad()
            G1_optim.zero_grad()
            G2_optim.zero_grad()
            D_optim.zero_grad()
            z_noise = torch.normal(0, 0.02, size=(BATCH_SIZE, INPUT_DIM)).to(device)
            x_fake_0 = G_0(z_noise)
            x_fake_1 = G_1(z_noise)
            x_fake_2 = G_2(z_noise)
            y_fake_0 = D(x_fake_0)
            y_fake_1 = D(x_fake_1)
            y_fake_2 = D(x_fake_2)
            ### Record mean and std of y_fake for further observation
            mean_yf0.append(torch.mean(y_fake_0).item())
            mean_yf1.append(torch.mean(y_fake_1).item())
            mean_yf2.append(torch.mean(y_fake_2).item())
            std_yf0.append(torch.std(y_fake_0).item())
            std_yf1.append(torch.std(y_fake_1).item())
            std_yf2.append(torch.std(y_fake_2).item())
            # concatenate 3 fakes
            # duplicate real
            x_fake = torch.cat((x_fake_0, x_fake_1, x_fake_2), 0)
            y_fake = torch.cat((y_fake_0, y_fake_1, y_fake_2), 0)
            x_real_x3 = x_real.repeat(3, 1, 1, 1)
            y_real = D(x_real_x3)
            alpha = torch.rand(x_real_x3.size()).to(device)
            interpolate = (torch.mul(alpha, (x_fake - x_real_x3)) + x_real_x3).to(device)
            y_interp = D(interpolate)
            gradient = autograd.grad(
                outputs = y_interp,
                inputs = interpolate,
                grad_outputs = torch.ones(y_interp.size()).to(device),
                create_graph = True,
                retain_graph = True,
                only_inputs = True)[0]
            gradient = gradient.flatten(1, 3)
            gradient = torch.mul(gradient, gradient)
            gradient = torch.sum(gradient, 1)
            gradient = torch.sqrt(gradient)
            k_val = torch.tensor(K).repeat(BATCH_SIZE, 1).to(device)
            gradient = gradient - k_val
            gradient = torch.mean(gradient ** 2)
            D_optim.zero_grad()
            loss_D = torch.mean(y_fake) - torch.mean(y_real) + LAMBDA * gradient
            loss_D.backward()
            D_optim.step()
    return mean_yf0, mean_yf1, mean_yf2, std_yf0, std_yf1, std_yf2



if __name__ == '__main__':
    print("Start")
    start = time.time()
    G_0 = Generator().to(device)
    G_1 = Generator().to(device)
    G_2 = Generator().to(device)
    D = Discriminator().to(device)
    
    if (args.contin > 0):
        G_0 = torch.load(CURRENT_WORK + "G0_" + str(args.contin) + ".pth")
        G_1 = torch.load(CURRENT_WORK + "G1_" + str(args.contin) + ".pth")
        G_2 = torch.load(CURRENT_WORK + "G2_" + str(args.contin) + ".pth")
        D = torch.load(CURRENT_WORK + "D_" + str(args.contin) + ".pth")
    
    G_0.train()
    G_1.train()
    G_2.train()
    D.train()
    optimizer_G0 = optim.Adam(G_0.parameters(), lr = LR, betas=(0.5, 0.9))
    optimizer_G1 = optim.Adam(G_1.parameters(), lr = LR, betas=(0.5, 0.9))
    optimizer_G2 = optim.Adam(G_2.parameters(), lr = LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr = LR, betas=(0.5, 0.9))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: (img - torch.amin(img)) / (torch.amax(img) - torch.amin(img)))
        ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_data,
        batch_size = BATCH_SIZE,
        shuffle = True
        )
    batch_count_train = int(len(train_data) / BATCH_SIZE) + 1
    
    mean_yfake_0 = []
    mean_yfake_1 = []
    mean_yfake_2 = []
    std_yfake_0 = []
    std_yfake_1 = []
    std_yfake_2 = []
    
    for epoch in num_epochs:
        print("epoch " + str(epoch))
        mean_yf0, mean_yf1, mean_yf2, std_yf0, std_yf1, std_yf2 = train(G_0, G_1, G_2, D, optimizer_G0, optimizer_G1, optimizer_G2, optimizer_D, train_loader)
        mean_yfake_0 = mean_yfake_0 + mean_yf0
        mean_yfake_1 = mean_yfake_1 + mean_yf1
        mean_yfake_2 = mean_yfake_2 + mean_yf2
        std_yfake_0 = std_yfake_0 + std_yf0
        std_yfake_1 = std_yfake_1 + std_yf1
        std_yfake_2 = std_yfake_2 + std_yf2
        torch.save(G_0, CURRENT_WORK + "G0_final.pth")
        torch.save(G_1, CURRENT_WORK + "G1_final.pth")
        torch.save(G_2, CURRENT_WORK + "G2_final.pth")
        torch.save(D, CURRENT_WORK + "D_final.pth")
        if (epoch in savelist):
            torch.save(G_0, CURRENT_WORK + "G0_" + str(epoch) + ".pth")
            torch.save(G_1, CURRENT_WORK + "G1_" + str(epoch) + ".pth")
            torch.save(G_2, CURRENT_WORK + "G2_" + str(epoch) + ".pth")
            torch.save(D, CURRENT_WORK + "D_" + str(epoch) + ".pth")
            
    
    mean_yfake_0 = np.asarray(mean_yfake_0)
    mean_yfake_1 = np.asarray(mean_yfake_1)
    mean_yfake_2 = np.asarray(mean_yfake_2)
    std_yfake_0 = np.asarray(std_yfake_0)
    std_yfake_1 = np.asarray(std_yfake_1)
    std_yfake_2 = np.asarray(std_yfake_2)
    
    np.save(CURRENT_WORK + "mean_yf0.npy", mean_yfake_0)
    np.save(CURRENT_WORK + "mean_yf1.npy", mean_yfake_1)
    np.save(CURRENT_WORK + "mean_yf2.npy", mean_yfake_2)
    np.save(CURRENT_WORK + "std_yf0.npy", std_yfake_0)
    np.save(CURRENT_WORK + "std_yf1.npy", std_yfake_1)
    np.save(CURRENT_WORK + "std_yf2.npy", std_yfake_2)
    
    
    end = time.time()
    print(end - start)
    