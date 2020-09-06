# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:04:20 2020

@author: kochark1
This program is to demonstrate DIP based reciever for mmWave MIMO channels


Configuration parameters:
    number_of_samples (int): This parameter contains the total number of
        samples that the experment considers.
    reciever_start_sample (int): This is an integer in 0 and number_of_samples
        (includs 0 and excluds number_of_samples). The reciever starts
        processing samples after some delay such that the data samples are
        ready for processing. reciever_start_sample represents this delay in
        sample.
    path (string): TODO
"""
# importing python packages
import threading
import time

# importing user defined modules
from shared_params import Simulation_parameters
from dataGen import DataGen
from reciever import Reciever

#------------------------------ Configurations ------------------------------#

# Time stamp recording
start = time.perf_counter()

# Default system parameters
# toDo

# overwrite parameters from shared_param file
number_of_samples = Simulation_parameters.number_of_samples
reciever_start_sample = Simulation_parameters.reciever_start_sample
threadingEnabled = Simulation_parameters.threadingEnabled
dataGenDisabled = Simulation_parameters.dataGenDisabled

# Derived parameters
dataGenEnabled = not dataGenDisabled
threadingDisabled = not threadingEnabled
waitedTooLong = (number_of_samples<=reciever_start_sample)

# Settting up results folder of dataGen part
Simulation_parameters.setup_results_folder()
results_folder_path = Simulation_parameters.results_folder_path

DataGen.setup_snr_folders()

# Configuring the reciever thread.
t = threading.Thread(target = Reciever,
                              args = [results_folder_path, number_of_samples])

# Time stamp recording and printing
intermediate = time.perf_counter()
print(f'Setup done in {round(intermediate-start, 2)} seconds(s)')

#--------------------------------- Main Code---------------------------------#

# Sample-wise processing.
for sample_id in range(number_of_samples):
    # If data need not be generated, start the reciever thread and break.
    if dataGenDisabled:
        if threadingEnabled:
            t.start()
        else:
            Reciever(results_folder_path, number_of_samples)
        break
    
    # Start the dataDatageneration sample by sample.
    DataGen(sample_id)
    Simulation_parameters.set_completionFlag(sample_id)
    # If configured, start reciever thread concurrently with the main thread.
    if threadingEnabled and (sample_id == reciever_start_sample):
        t.start()

# Handling the case of missing t.start()
if dataGenEnabled and (threadingDisabled or waitedTooLong):
        Reciever(results_folder_path, number_of_samples)
elif dataGenDisabled and threadingEnabled:
    # Wait till the reciever thread is done for accurate total time.
    t.join()

# Compute and display execution time.
finish = time.perf_counter()
print(f'Finished in {round(finish-intermediate, 2)} seconds(s)')