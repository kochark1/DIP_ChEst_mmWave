# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:41:11 2020

@author: kochark1
"""
import numpy as np
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
        # To be replaced by the user code
        pass
    
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
        # To be replaced by the user code
        return np.random.uniform(0,1,1)
    
    # more methods can be implemented
    
class Plotter_and_analyzer:
    def __init__(self, system_parameters_passed, channel_parameters_passed):
        # To be replaced by the user code
        pass
    
    def nmse_plotter(self, performance_results):
        # To be replaced by the user code
        pass
    
    # more methods can be implemented