"""

@author: Pavithra Vijayakrishnan

"""

from deep_channel_estimator import deep_channel_estimator as DCE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DIP_training:
    def __init__(self,Y_input, layers=6, out_channel_opt= 8 ,SNR = 0, lr = 0.01):
        
        self.layers = layers 
        self.out_channel_opt = out_channel_opt
        self.SNR = SNR 
        self.Y_input = Y_input
        self.lr = lr
        return 
    
    def training(self,device):
        
        Y_input = self.Y_input
        Y_input = np.transpose(Y_input , (3,0,1,2))
        snr_val = self.SNR
        snr_lin_val = pow(10,(snr_val/10))
        user_samples = Y_input.shape[0] 
        epoch_dict = {'M2_8':2000,'M2_16':1300,'M2_32':900,'M2_64':250}
        M =  Y_input.shape[1]
        layers = self.layers #Can configure this in the new systems file if needed
        out_channel = self.out_channel_opt
        MM = Y_input.shape[2] # No of receiver antennas in mmWave
        NN = Y_input.shape[3] # No of Transmitter antennas in mmWave
        MM_den = pow(2,layers-1)
        NN_den = pow(2,layers-1)
        Z0 = np.random.rand(user_samples,out_channel,int(MM/MM_den),int(NN/NN_den))
        Z1 = torch.from_numpy(Z0)
        Z1.type(torch.DoubleTensor)
        in_channel = Y_input.shape[1]
        results = []
        result_orig = []
        max_epoch = epoch_dict['M'+str(M)+'_'+str(out_channel)]
        Final_vals = []
        dce = DCE(in_channel,out_channel,layers)
        dce.to(device)
        optimizer = optim.Adam(dce.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()
        inp = Y_input[0,:,:,:]
        inputCPU = Z1[0,:,:,:].float()
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
                
            loss = mse(val,torch.unsqueeze(inp, 0).float())

            ll = loss.item()
            print("Loss at epoch ,j," is ", ll) 
            loss.backward()
            optimizer.step()
            
        y_output = val
        y_output = y_output.cpu().detach().numpy()
        return y_output

      
