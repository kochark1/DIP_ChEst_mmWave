# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:41:11 2020

@author: kochark1
"""
import numpy as np
import matplotlib.pyplot as plt

import os.path
import os
from DIP_training import DIP_training as DIP

class ChannelEstimator:
    """
    The class is dedicated to include user defined functionality for the
    reciever module.
    Ex:
        Channel Estimation
        Datapath (equalization, demodulation and decoding)
    
    """
    def __init__(self, system_parameters_passed, channel_parameters_passed,
                 snr_id):
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
        self.snr_id = snr_id
    
    def estimate(self, y_matrix, h_matrix_org):
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
            snr_dB = self.system_parameters.snr_dB[self.snr_id]
            snr_lin = 10**(snr_dB/10)
            h_LS = np.matmul(y_matrix,
                             np.linalg.pinv(self.system_parameters.t_matrix))
            h_LS = np.sqrt((snr_lin+1)/snr_lin)*h_LS
            h_matrix = h_LS
            h_matrix = dip_processing(h_LS)
            return h_matrix
        
        def nmse_calculator(h_est, h_org):
            mse = np.linalg.norm((h_est - h_org), 'fro')
            den = np.linalg.norm(h_org, 'fro')
            if den <= (1e-7):
                return mse*1e7
            return mse/den
        
        def dip_processing(Y_input):
            """
            Parameters:
            
                Y_input - matrix of size 64 * 64
                
                out_channel_list - Hyper parameter that represent list of
                number of channels for the CNN eg:[8]or[8,16,32]
                
                SNR - integer that represents the SNR of the signal eg: 0 or 5
                
                layers - Hyper parameter : Number of hidden layers for the
                network. Integer should be more than 1
                
                lr - Hyper parameter : Learning rate for the optimizing model.
                Default is 0.01
            
            DESCRIPTION: This function invokes the functions to perform DIP
            estimation to denoise the Y_input for the given SNR by
            experimenting over the specified set of hyper parameters like
            layers, out_channel_list and lr

            Returns:
                A dictionary Y_output of the form
                    Y_output = {'SNR5_k8' : Y_out, ...}
                where the key is made up of SNR and each of the number from
                out_channel_list and the corresponding Y_out is the denoised
                signal for the respective SNR and out_channel_opt value
        
            """
           
            # Complex matrix Y_input of size 64*64 split to form
            # 2*64*64*1 real matrix
            # We are training or fitting 1 sample at a time unlike the
            # DNN training, so the dim = 3 has size 1
            
            Y_input_DIP = np.zeros((2, Y_input.shape[0], Y_input.shape[1]))
            Y_input_DIP[0,:,:] =  Y_input.real.astype(np.float)
            Y_input_DIP[1,:,:] =  Y_input.imag.astype(np.float)
            
            mn_real = np.mean(Y_input_DIP[0,:,:])
            mn_imag = np.mean(Y_input_DIP[1,:,:])
            mn = mn_real + 1j*mn_imag
            
            Y_input_DIP[0,:,:] -= mn_real
            Y_input_DIP[1,:,:] -= mn_imag
            norm_factor = 2*np.amax(abs(Y_input_DIP))
            Y_input_DIP /= norm_factor
            Y_input_DIP += 0.5
            
            DIP_training_1 = DIP(Y_input_DIP)
            Y_temp = DIP_training_1.training()
            
            Y_output = Y_temp[0,:,:] + 1j* Y_temp[1,:,:]
            Y_output -= 0.5*(1+1j)
            Y_output *= norm_factor
            Y_output += mn
     
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
        plt.plot(snr_dB, 10*np.log10(performance_results))
        plt.draw()
        if not os.path.exists('Results'):
               os.makedirs('Results')
        plt.savefig('Results/plot.png')
        np.save('Results/nmse_vals.npy', performance_results)
