# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:48:13 2020

@author: kochark1
"""
import numpy as np
from scipy.sparse import random
from scipy.linalg import dft, circulant
from math import ceil

def quantize(y, b):
    del_b = 2**(-b)
    levels = 2**(-(b-1))
    del_real = (1/(2**6))*del_b
    del_imag = (1/(2**6))*del_b
    y_out = 1j*np.zeros(y.shape)
    for ii in range(len(y[:,0])):
        for jj in range(len(y[0,:])):
            z = y[ii,jj]
            q_real = min(ceil(abs(z.real)/del_real), levels)-1/2
            q_imag = min(ceil(abs(z.imag)/del_imag), levels)-1/2
            real_val =  (np.sign(z.real))*q_real*del_real
            imag_val =  (np.sign(z.imag))*q_imag*del_imag
            y_out[ii,jj] = real_val + 1j*imag_val
    return y_out
    

class SystemModel:
    """
    The class is dedicated to include user defined functionality for the data
    generation.
    
    """
    den = 0.0005
    
    def __init__(self, snr_id):
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
        self.snr_id = snr_id
    def generate(self):
        x_matrix_org = self.generateChannel()
        y_matrix = self.generateTXsignal(x_matrix_org)
        x_temp = np.matmul(SystemModel.b_left_matrix, x_matrix_org)
        h_matrix_org = np.matmul(x_temp, SystemModel.b_nt_matrix.conj().T)
        return [y_matrix, h_matrix_org]
    
    def generateChannel(self):
        nT = self.system_parameters.nT
        nR = self.system_parameters.nR
        x_matrix = 1j*(np.zeros((nR, nT *self.system_parameters.const_L)))
        for l in range(self.system_parameters.const_L):
            mask_mat = random(nR, nT, density = SystemModel.den).A
            mask_mat = (abs(mask_mat)>1e-4)
            real_mat = mask_mat*np.random.randn(nR,nT)*np.sqrt(1/2)
            imag_mat = mask_mat*np.random.randn(nR,nT)*np.sqrt(1/2)
            x_matrix[:, (l*nT):((l+1)*nT)] = real_mat + 1j * imag_mat
        x_matrix /= np.sqrt(SystemModel.den*nT*nR)
        return x_matrix
    def generateTXsignal(self, h_matrix):
        nR = self.system_parameters.nR
        nP = self.system_parameters.nP
        
        snr_dB = self.system_parameters.snr_dB[self.snr_id]
        snr_lin = 10**(snr_dB/10)
        b_left_matrix = SystemModel.b_left_matrix
        b_right_matrix = SystemModel.b_right_matrix
        z_matrix = np.matmul(np.matmul(b_left_matrix, h_matrix),
                             b_right_matrix)
        noise_matrix = (1/np.sqrt(2))*np.random.randn(nR, nP) +\
            1j * (1/np.sqrt(2))*np.random.randn(nR, nP)
        noise_matrix /= np.sqrt(nR*nP)
        y_matrix = np.sqrt(snr_lin/(snr_lin+1))*z_matrix +\
            np.sqrt(1/(snr_lin+1))*noise_matrix
        # y_matrix = quantize(y_matrix, 1)
        return y_matrix
    
    @classmethod
    def set_config(cls, system_parameters_passed, channel_parameters_passed):
        cls.system_parameters = system_parameters_passed
        cls.channel_parameters = channel_parameters_passed
        cls.recieveBeamgenerator()
        cls.right_matrix_constructor()
            
    @classmethod
    def recieveBeamgenerator(cls):
        nR_h = cls.system_parameters.nR_h
        nR_v = cls.system_parameters.nR_v
        
        F_h =  dft(nR_h)/np.sqrt(nR_h)
        F_v =  dft(nR_v)/np.sqrt(nR_v)
        cls.b_left_matrix = np.kron(F_v, F_h)
        
    @classmethod
    def trasnmitBeamgenerator(cls):
        nT_h = cls.system_parameters.nT_h
        nT_v = cls.system_parameters.nT_v
        
        F_h =  dft(nT_h)/np.sqrt(nT_h)
        F_v =  dft(nT_v)/np.sqrt(nT_v)
        cls.b_nt_matrix = np.kron(F_v, F_h)
        return np.kron(F_v, F_h)
    
    @classmethod
    def right_matrix_constructor(cls):
        b_nt_matrix = SystemModel.trasnmitBeamgenerator()
        nP = cls.system_parameters.nP
        nT = cls.system_parameters.nT
        first_row_zeroshift = np.zeros((nP,))
        first_row_zeroshift[0] = 1.0
        b_matrix = 1j*np.zeros((nT *cls.system_parameters.const_L, nP))
        bt_matrix = np.matmul(b_nt_matrix.conj().T,
                              cls.system_parameters.t_matrix)
        for l in range(cls.system_parameters.const_L):
            first_row_shifted = np.roll(first_row_zeroshift, l)
            j_l_matrix = circulant(first_row_shifted)
            b_matrix[(l*nT):((l+1)*nT), :] = np.matmul(bt_matrix, j_l_matrix)
        
        cls.b_right_matrix = b_matrix