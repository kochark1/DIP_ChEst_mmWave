# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:41:56 2020

@author: kochark1
This file includes set of parameters shared by all .py files in the project.
"""
import os
import shutil
import time
import sys
import numpy as np
class Simulation_parameters:
    dataGenDisabled = False
    threadingEnabled = False
    number_of_samples = 5
    reciever_start_sample = 4
    results_folder = "dataSet"
    completionFlag = np.zeros((number_of_samples, 1))
    @classmethod
    def set_completionFlag(cls, sample_id):
        cls.completionFlag[sample_id] = 1
    @classmethod
    def setup_results_folder(cls):
        cls.root_path = os.getcwd()
        cls.results_folder_path = os.path.join(cls.root_path,
                                               cls.results_folder)
        if cls.dataGenDisabled:
            return
        itr = 0
        while os.path.exists(cls.results_folder) and itr<5:
            shutil.rmtree(cls.results_folder, ignore_errors=False,
                          onerror=None)
            time.sleep(0.5)
            itr+=1
        if os.path.exists(cls.results_folder):
            print("\n'dataSet' folder was not deleted")
            sys.exit()
        os.mkdir(cls.results_folder)

class System_parameters:
    nT_h = 8 # number of horixontal antennas at the Tx
    nT_v = 8 # number of vertical antennas at the Tx
    nT = nT_v * nT_h # Number_of_Tx_antennas
    
    nR_h = 8 # number of horixontal antennas at the Rx
    nR_v = 8 # number of vertical antennas at the Rx
    nR = nR_v * nR_h # Number_of_Rx_antennas
    const_L = 1 # ISI spread due to pulse shaping
    fC = 28e9 # carrier frequency in Hz
    nP = 2048 # number of pilot symbols
    snr_dB = range(-10,45,5)
    pilot_structure = 'Random'
    
    @classmethod
    def set_class_config(cls):
        nT = System_parameters.nT
        nP = System_parameters.nP
        if System_parameters.pilot_structure == 'Random':
            cls.t_matrix = np.random.randn(nT,nP) +\
                np.random.randn(nT,nP) * 1j
            cls.t_matrix = np.sqrt(1/(2*nP))*cls.t_matrix
        else:
            print(f"""System parameter setting error: pilot structure cannot
                  be {System_parameters.pilot_structure}""")
        
        os.chdir(Simulation_parameters.results_folder_path)
        cls.snr_folder_list = []
        for snr in System_parameters.snr_dB:
            if snr>=0:
                snr_folder = f'snr{snr}'
            else:
                snr_folder = f'snr_minus{-snr}'
            os.mkdir(snr_folder)
            cls.snr_folder_list.append(snr_folder)
        os.chdir(Simulation_parameters.root_path)

class Channel_parameters:
    number_of_clusters = 4
    number_of_paths = 10
    azmuth_spread_dg = 7.5 # in degrees
    elevation_spread_dg = 7.5 # in degrees

    max_delay_spread = 16 # in samples