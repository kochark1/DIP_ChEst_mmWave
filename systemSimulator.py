# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:48:13 2020

@author: kochark1
"""
import numpy as np
class SystemModel:
    """
    The class is dedicated to include user defined functionality for the data
    generation.
    
    """
    def __init__(self, System_parameters_passed, Channel_parameters_passed):
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
        y_matrix : Final datasample for a given setup and SNR value.

        """
        self.system_parameters = System_parameters_passed
        self.channel_parameters = Channel_parameters_passed
    def generate(self, snr_lin):
        # Replace your code
        h_matrix_org =  np.random.randn(self.system_parameters.nR,
                            self.system_parameters.nT)
        y_matrix = np.random.randn(self.system_parameters.nT,
                            self.system_parameters.nP)
        return [y_matrix, h_matrix_org]