from deep_channel_estimator import deep_channel_estimator as DCE
import numpy as np
import math

import time
from numpy import linalg as LA
import scipy.io
import copy
from pylab import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os.path
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
class DIP_training:
    def __init__(self,Y_input, layers=6, out_channel_opt=[8],SNR=0,lr = 0.01):
        self.layers = layers #int
        self.out_channel_opt = out_channel_opt # list
        self.SNR = SNR #int
        self.Y_input = Y_input
        self.lr = lr
    def training(self,device):

        Y_input = self.Y_input
        Y_input = np.transpose(Y_input , (3,0,1,2))
        snr_val = self.SNR
        snr_lin_val = pow(10,(snr_val/10))
        user_samples = Y_input.shape[0] 
        epoch_dict = {'M1_8':2000,'M1_16':1300,'M1_32':900,'M1_64':250,'M64_8':4000,'M64_16':1970,'M64_32':1800,'M64_64':1000}
        M = 1
        for m in range(len(self.out_channel_opt)):

            layers = self.layers #Can configure this in the new systems file if needed
            out_channel = self.out_channel_opt[m]
            MM = Y_input.shape[2] # No of receiver antennas in mmWave
            NN = Y_input.shape[3] # No of Transmitter antennas in mmWave
            MM_den = pow(2,layers-1)
            NN_den = pow(2,layers-1)
            Z0 = np.random.rand(user_samples,out_channel,int(MM/MM_den),int(NN/NN_den))
            print("Size of the input random tensor is ", Z0.shape)
            Z1 = torch.from_numpy(Z0)
            Z1.type(torch.DoubleTensor)
             
            in_channel = 2* M
            results = []
            result_orig = []

            max_epoch = epoch_dict['M'+str(M)+'_'+str(out_channel)]
            print('MAX_EPOCH is ',max_epoch,' for M and out channel as ',str(M), str(out_channel))

            for k in range(0,user_samples):
                Final_vals = []
                dce = DCE(in_channel,out_channel,layers)
                dce.to(device)
                optimizer = optim.Adam(dce.parameters(), lr=self.lr)
                mse = torch.nn.MSELoss()
                inp = Y_input[k,:,:,:]

                inputCPU = Z1[k,:,:,:].float()
                inputGPU = inputCPU.to(device)
                avg_Denoised = []
                ll_check = []
                for j in range(max_epoch):
                    print("Inside training epoch ",j)
                    optimizer.zero_grad()

                    try:
                        val = dce.forward(inputGPU)

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print("Heyy you here ")
                            sys.stdout.flush()
                    	    #for p in dce.parameters():
                             #   if p.grad is not None:
                              #      del p.grad

                        else:
                            raise e
                print("Outside forward class ")
                loss = mse(val,torch.unsqueeze(inp, 0).float())

                ll = loss.item()
                print("loss at epoch ",j , " is ",ll)
                loss.backward()
                optimizer.step()

                val1 = val;
                val1 = val1.cpu().detach().numpy()


            str3 = "i is "+str(i)+" k is "+str(k)+ "   out channel is "+str(out_channel)+'\n'
            Final_vals.append([i,m,k,j,val1])
            str0 = "Loss is "+str(ll)+'\n'
            print(str0)

            L = [str3,str0]
            file1.writelines(L)



            save_Path = 'Results/M'+str(M)+'/SNR_'+str(SNR[i])+'/k'+str(out_channel)

            if not os.path.exists(save_Path):
                    os.makedirs(save_Path)
                    print("Directory " , save_Path ,  " Created ")
            else:
                    print("Directory " , save_Path ,  " already exists")

            np.save(save_Path+'/Res_Final_vals_'+str(k)+'.npy',Final_vals)

