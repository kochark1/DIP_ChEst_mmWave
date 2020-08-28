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
        
        def dip_processing(Y_input):
            GPU = True
            if GPU == True and torch.cuda.ids_available():
                device = torch.device('cuda:0')
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
            else:
                device = torch.device('cpu')
           
            # Complex matrix Y_input of size 64*64 split to form
            # 2*64*64*1 real matrix
            
            Y_input_DIP = np.zeros(2,Y_input.shape[0],Y_input.shape[1],1)
            Y_input_DIP[0,:,:,:] =  Y_input.real.astype(np.float)
            Y_input_DIP[1,:,:,:] =  Y_input.imag.astype(np.float)
           
            
            # ToDo Pavithra task 2: clean up inside two files as much as
            # possible
            DIP_training_1 = DIP(Y_input)
            Y_output = DIP_training_1.training(device)
            # ToDo Pavithra task 3: read Y_output from DIP_training_1 rather
            # than saving into a file
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
