# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:41:11 2020

@author: kochark1
"""
import numpy as np
import matplotlib.pyplot as plt

from DIP_training import DIP_training as DIP

class ChannelEstimator:
    """
    The class is dedicated to include user defined functionality for the
    reciever module.
    Ex:
        Channel Estimation
        Datapath (equalization, demodulation and decoding)
    
    """
    def __init__(self, system_parameters_passed, channel_parameters_passed):
        """
        

        Parameters
        ----------
        System_parameters : class
            All the parameters defined in System_parameters in
            shared_params.py.
        Channel_parameters : class
            All the parameters defined in Channel_parameters in
            shared_params.py.

        Returns
        -------
        None.

        """
        self.system_parameters = system_parameters_passed
        self.channel_parameters = channel_parameters_passed
    
    def estimate(self, y_matrix, h_matrix_org, snr_id):
        """
        

        Parameters
        ----------
        snr_id : int
            DESCRIPTION.

        Returns
        -------
        None

        """
        def channel_estimation(self, y_matrix):
            nT = self.system_parameters.nT
            nR = self.system_parameters.nR
            const_L = self.system_parameters.const_L
            h_LS = np.matmul(y_matrix,
                             np.linalg.pinv(self.system_parameters.t_matrix))
            h_matrix = dip_processing(h_LS)
            return h_matrix
        
        def nmse_calculator(h_est, h_org):
            mse = np.mean((abs(h_est - h_org))**2)
            den = np.mean((abs(h_org))**2)
            if den <= (1e-7):
                return mse*1e7
            return mse/den
        
        def dip_processing(Y_input, out_channel_list, SNR, layers = 6, lr = 0.01 ):
            """
            Parameters:
            Y_input - matrix of size 64 * 64
            out_channel_list - Hyper parameter that represent list of number of channels for the CNN eg:[8]or[8,16,32]
            SNR - integer that represents the SNR of th e signal eg: 0 or 5
            layers - Hyper parameter : Number of hidden layers for the network. Integer should be more than 1
            lr - Hyper parameter : Learning rate for the optimizing model. Default is 0.01
            
            DESCRIPTION: This function invokes the functions to perform DIP estimation to denoise the Y_input 
            for the given SNR by experimenting over the specified set of hyper parameters like layers, out_channel_list
            and lr

            Returns:
            A dictionary Y_output of the form 
                    Y_output = {'SNR5'_k8' : Y_out, ...}
            where the key is made up of SNR and each of the number from out_channel_list and the corresponding Y_out 
            is the denoised signal for the respective SNR and out_channel_opt value
        
            """
            GPU = True
            
            if GPU == True and torch.cuda.ids_available():
                device = torch.device('cuda:0')
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
            else:
                device = torch.device('cpu')
           
            # Complex matrix Y_input of size 64*64 split to form
            # 2*64*64*1 real matrix
            # We are training or fitting 1 sample at a time unlike the DNN training, so the dim = 3 has size 1
            
            Y_input_DIP = np.zeros(2,Y_input.shape[0],Y_input.shape[1],1)
            Y_input_DIP[0,:,:,:] =  Y_input.real.astype(np.float)
            Y_input_DIP[1,:,:,:] =  Y_input.imag.astype(np.float)
            
            Y_output = {}
            for i in out_channel_list:
                DIP_training_1 = DIP(Y_input,layers, out_channel_opt= i ,SNR, lr)
                Y_output['SNR'+str(SNR)+'_k'+str(i)] = DIP_training_1.training(device)
     
            return Y_output
        
        h_matrix_estimate = channel_estimation(self, y_matrix)
        nmse_error = nmse_calculator(h_matrix_estimate, h_matrix_org)
        return nmse_error
    
    # more methods can be implemented
    
class Plotter_and_analyzer:
    @classmethod
    def set_config(cls, system_parameters_passed):
        cls.system_parameters = system_parameters_passed
    
    def nmse_plotter(self, performance_results):
        snr_dB = Plotter_and_analyzer.system_parameters.snr_dB
        plt.plot(snr_dB, performance_results)
        plt.draw()
