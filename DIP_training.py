"""

@author: Pavithra Vijayakrishnan

"""

from deep_channel_estimator import deep_channel_estimator as DCE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys


class DIP_training:
    layers = 6
    out_channel_opt = 2
    lr = 0.01
    # epoch_dict = {'M2_8':2000,'M2_16':1300,'M2_32':900,'M2_64':250}
    max_epoch = 500
    
    def __init__(self, Y_input):
        
        self.Y_input = Y_input
        GPU = True
        if GPU == True and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        
        return 
    
    def training(self):
        Y_input = self.Y_input
        device = self.device
        M =  Y_input.shape[0]
        layers = self.layers #Can configure this in the new systems file if needed
        out_channel = self.out_channel_opt
        MM = Y_input.shape[1] # No of receiver antennas in mmWave
        NN = Y_input.shape[2] # No of Transmitter antennas in mmWave
        MM_den = pow(2,layers-1)
        NN_den = pow(2,layers-1)
        Z0 = np.random.rand(out_channel,int(MM/MM_den),int(NN/NN_den))
        Z1 = torch.from_numpy(Z0)
        Z1.type(torch.DoubleTensor)
        in_channel = Y_input.shape[0]
        results = []
        result_orig = []
        max_epoch = self.max_epoch
        Final_vals = []
        dce = DCE(in_channel,out_channel,layers)
        dce.to(device)
        optimizer = optim.Adam(dce.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()
        inp = np.zeros((1,) + Y_input.shape)
        inp[0,:,:,:] = Y_input
        inp_torch = torch.from_numpy(inp)
        inp = inp_torch.to(device)
        inputCPU = Z1.float()
        inputGPU = inputCPU.to(device)
        avg_Denoised = []
        ll_check = []
        for j in range(max_epoch):
             
            optimizer.zero_grad()

            try:
                val = dce.forward(inputGPU)

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    sys.stdout.flush()
                else:
                    raise e
                
            loss = mse(val, inp.float())

            ll = loss.item()
            #print("Loss at epoch" ,j," is ", ll) 
            loss.backward()
            optimizer.step()
            
        y_output = val[0,:,:,:]
        y_output = y_output.cpu().detach().numpy()
        return y_output

      
