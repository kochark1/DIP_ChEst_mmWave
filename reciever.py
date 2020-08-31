# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:32:40 2020

@author: kochark1
"""
import numpy as np
import os
import threading
import time

from shared_params import Simulation_parameters, System_parameters,\
    Channel_parameters
from reciever_operations import ChannelEstimator, Plotter_and_analyzer
class Reciever:
    def __init__(self, path, number_of_samples):
        self.path = path
        self.number_of_samples = number_of_samples
        Plotter_and_analyzer.set_config(System_parameters)
        plotter_and_analyzer = Plotter_and_analyzer()
        snr_size = len(System_parameters.snr_dB)
        avg_nmse = np.zeros((snr_size,1))
        for sample_id in range(self.number_of_samples):
            tt = 0
            while not Simulation_parameters.completionFlag[sample_id]:
                time.sleep(0.5)
                tt += 1
            
            threads = []
            performance_results = np.zeros((len(System_parameters.snr_dB), 1))
            snr_id = 0
            for snr_folder in System_parameters.snr_folder_list:
                filePath =\
                    os.path.join(Simulation_parameters.results_folder_path,
                                 snr_folder)
                if Simulation_parameters.threadingEnabled:
                    t = threading.Thread(target=self.snr_branch_estimator,
                                         args=[filePath, sample_id, snr_id])
                    t.daemon = True
                    t.start()
                    threads = threads + [t,]
                else:
                    results_temp = self.snr_branch_estimator(filePath,
                                                             sample_id,
                                                             snr_id)
                    performance_results[snr_id] = results_temp
                snr_id += 1
            
            if Simulation_parameters.threadingEnabled:
                snr_id = 0
                for thread in threads:
                    performance_results[snr_id] = thread.join()
                    snr_id += 1
            print(f"Sample {sample_id} done!{performance_results.T}")
            avg_nmse += ((1/self.number_of_samples)*performance_results)
        plotter_and_analyzer.nmse_plotter(avg_nmse)
            
    
    def snr_branch_estimator(self, filePath, sample_id, snr_id):
        filePathAndName = os.path.join(filePath, f'Y_sample{sample_id}.npy')
        y_matrix = np.load(filePathAndName)
        
        
        filePathAndName = os.path.join(filePath, f'H_sample{sample_id}.npy')
        h_matrix_org = np.load(filePathAndName)
        
        # can develop more classes in reciever_operations and use based on
        # requirement
        channelEstimator = ChannelEstimator(System_parameters,
                                            Channel_parameters)
        return channelEstimator.estimate(y_matrix, h_matrix_org, snr_id)