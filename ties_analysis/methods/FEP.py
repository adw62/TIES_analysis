#!/usr/bin/env python

import numpy as np
import copy
import os
from pathlib import Path

from pymbar import MBAR, timeseries

from ties_analysis.methods.TI import compute_bs_error

#GLOBAL CONSTANTS
KBT = 0.5961619840741075 #unit kilocalorie/mole assumed 300k

class MBAR_Analysis():
    '''
    Class for MBAR analysis
    '''

    def __init__(self, MBARs, lambdas, analysis_dir):
        '''

        :param MBARs: numpy array for all results
        :param lambdas: Lambda class containing schedule
        :param analysis_dir, string file path for where to save analysis output
        '''

        self.data = MBARs

        self.shape = list(self.data.shape)
        self.nstates = self.shape[1]
        print('Data shape is {} repeats, {} states_i, {} states_j, {} iterations'.format(*self.shape))

        self.lambdas = lambdas

        self.analysis_dir = analysis_dir
        if analysis_dir is not None:
            print('Attempting to make analysis folder {0}'.format(self.analysis_dir))
            Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)

        self.data, self.N_k, self.rep_data, self.rep_N_k = MBAR_Analysis.decorrelate_data(self)

    def decorrelate_data(self):
        '''

        decorrolate time series data.
        :return turple, (np matrix containing decorrelated data, list of ints for end of decorrelated data in matrix)
        '''

        #load data if avaliable
        if self.analysis_dir is not None:
            decorr_data_sav = os.path.join(self.analysis_dir, 'fep_decorr_data.npy')
            index_sav = os.path.join(self.analysis_dir, 'fep_N_k.npy')

            rep_decorr_data_sav = os.path.join(self.analysis_dir, 'fep_rep_decorr_data.npy')
            rep_index_sav = os.path.join(self.analysis_dir, 'fep_rep_N_k.npy')

            if os.path.exists(decorr_data_sav) and os.path.exists(index_sav)\
                    and os.path.exists(rep_decorr_data_sav) and os.path.exists(rep_index_sav):

                print('Loading decorrelated data from disk')
                decorr_data = np.load(decorr_data_sav)
                N_k = np.load(index_sav)

                replicas_deccor = np.load(rep_decorr_data_sav)
                replicas_Nk = np.load(rep_index_sav)
                return decorr_data, N_k, replicas_deccor, replicas_Nk

        print('Decorrelating data...')
        iter_per_rep = self.shape[3]
        num_repeats = self.shape[0]
        tot_iter = iter_per_rep * num_repeats

        # decorr data matrix is square for numpy convenience and padded with zeros.
        # N_k allows know were the data ends and the padding begins in the matrix.
        N_k = np.zeros([self.nstates], np.int32)
        decorr_data = np.zeros([self.nstates, self.nstates, tot_iter])

        # to compute ensemble stdev we also would like to save the result for each replica separately
        replicas_deccor = []
        replicas_Nk = []
        blank_decorr_data = np.zeros([self.nstates, self.nstates, iter_per_rep])
        blank_Nk = np.zeros([self.nstates], np.int32)

        for i, u_kln in enumerate(self.data):
            rep_deccor_data = copy.deepcopy(blank_decorr_data)
            rep_Nk = copy.deepcopy(blank_Nk)
            for k in range(self.nstates):
                [nequil, g, Neff_max] = timeseries.detectEquilibration(u_kln[k, k, :])
                sub_idx = timeseries.subsampleCorrelatedData(u_kln[k, k, :], g=g)
                decorr_data[k, :, 0 + N_k[k]:N_k[k] + len(sub_idx)] = u_kln[k, :, sub_idx].T
                rep_deccor_data[k, :, 0:len(sub_idx)] = u_kln[k, :, sub_idx].T
                N_k[k] += len(sub_idx)
                rep_Nk[k] = len(sub_idx)
            replicas_deccor.append(rep_deccor_data)
            replicas_Nk.append(rep_Nk)

        print('Number of decorrelated samples extracted per state: {}'.format(N_k))

        if self.analysis_dir is not None:
            np.save(decorr_data_sav, decorr_data)
            np.save(index_sav, N_k)

            np.save(rep_decorr_data_sav, replicas_deccor)
            np.save(rep_index_sav, replicas_Nk)

        return decorr_data, N_k, replicas_deccor, replicas_Nk

    def replica_analysis(self, mask_windows=None):
        '''
        Function to make analysis of result from MBAR considering each trajectory as one replica
        :param mask_windows: list of ints, can be used to specify what windows to remove
        :return: list, containing average of bootstrapped dG and SEM
        '''

        replica_results = []
        for d, n in zip(self.rep_data, self.rep_N_k):
            results = self.analysis(u_kln=d, N_k=n, mask_windows=mask_windows)
            replica_results.append(results)

        # x[1] here is MBAR error which is not used
        dg = [x[0] for x in replica_results]
        #print('dG for each replica')
        #print(dg)

        result = compute_bs_error(dg)
        return [result[0], np.sqrt(result[1])]

    def analysis(self, u_kln=None, N_k=None, mask_windows=None):
        '''
        Process a matrix of potentials passes to this function as u_kln or process self.data which
         is matrix of all replicas
        :param u_kln: numpy array matrix of potentials for one replica
        :param N_k: list of ints, specifies which entry in u_kln should be read up to for each window
        :param mask_windows: list of ints, can be used to specify what windows to remove from u_kln
        :return: list, containing dG and associated error calculated by MBAR
        '''

        if u_kln is None:
            u_kln = self.data
        if N_k is None:
            N_k = self.N_k

        # remove windows
        if mask_windows is not None:
            mask_windows.sort()
            mask_windows.reverse()
            print('Removing windows {}'.format(mask_windows))
            for win in mask_windows:
                u_kln = np.delete(u_kln, win, axis=0)
                u_kln = np.delete(u_kln, win, axis=1)
                N_k = np.delete(N_k, win, axis=0)

        # Compute free energy differences and statistical uncertainties
        mbar = MBAR(u_kln, N_k)
        [DeltaF_ij, dDeltaF_ij, _] = mbar.getFreeEnergyDifferences()

        # print("Number of uncorrelated samples per state: {}".format(N_k))
        result = ([DeltaF_ij[0, len(N_k) - 1]*KBT,
                   dDeltaF_ij[0, len(N_k) - 1]*KBT]) #units kilocalorie per mol

        # Save data to analysis dir
        if self.analysis_dir is not None:
            np.save(os.path.join(self.analysis_dir, 'DeltaF_ij.npy'), DeltaF_ij)
            np.save(os.path.join(self.analysis_dir, 'dDeltaF_ij.npy'), dDeltaF_ij)

        return result