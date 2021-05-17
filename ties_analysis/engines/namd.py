#!/usr/bin/env python


import os
import glob
import collections
import numpy as np

from engines.openmm import Lambdas
from methods.TI import TI_Analysis

class NAMD(object):
    '''
    Class to perform TIES analysis on NAMD results
    '''
    def __init__(self, method, output, win_mask, namd_version, vdw_a, vdw_d, ele_a, ele_d, iterations):
        '''
        :param method: str, 'TI' or 'FEP'
        :param output: str, pointing to base dir of where output will be writen
        :param win_mask: list of ints, what windows if any to remove from analysis
        :param namd_version: float, for which version of NAMD generated alch files
        :param vdw_a: list of floats, describes lambda schedule for vdw appear
        :param vdw_d: list of floats, describes lambda schedule for vdw disappear
        :param ele_a: list of floats, describes lambda schedule for elec appear
        :param ele_d: list of floats, describes lambda schedule for elec disappear
        :param iterations: int, number of iterations to expect in alch files
        '''

        self.namd_ver = float(namd_version[0])

        if self.namd_ver < 3:
            self.name = 'N2'
        else:
            self.name = 'N3'

        self.iter = int(iterations[0])
        self.method = method
        self.namd_lambs = Lambdas(vdw_a, vdw_d, ele_a, ele_d)
        self.output = output
        self.win_mask = win_mask

    def run_analysis(self,  data_root, prot, lig, leg):
        '''

        :param data_root: str, file path point to base dir for results files
        :param prot: str, name of dir for protein
        :param lig:  str, name of dir for ligand
        :param leg: str, name of dir for thermo leg
        :return: list of floats, [dg, stdev(dg)]
        '''

        if self.method == 'FEP':
            raise NotImplemented()

        data = self.collate_data(data_root, prot, lig, leg)
        analysis_dir = os.path.join(self.output, self.name, self.method, prot, lig, leg)

        method_run = TI_Analysis(data, self.namd_lambs, analysis_dir)

        result = method_run.analysis(mask_windows=self.win_mask)

        return result

    def collate_data(self, data_root, prot, lig, leg):
        '''
        Function to iterate over replica and window dirs reading NAMD alch file and building numpy array of potentials
        :param data_root: str, file path point to base dir for results files
        :param prot: str, name of dir for protein
        :param lig:  str, name of dir for ligand
        :param leg: str, name of dir for thermo leg
        :return: numpy.array for all potential collected
        '''

        results_dir_path = os.path.join(data_root, self.name, self.method, prot, lig, leg)

        result_files = os.path.join(results_dir_path, 'LAMBDA_*', 'rep*', 'simulation', 'sim1.alch')
        result_files = list(glob.iglob(result_files))

        # Sort by order of windows
        result_files.sort(key=get_window)

        #print('Processing files...')
        #for file in result_files:
        #    print(file)

        # Use ordered dict to preserve windows order
        all_data = collections.OrderedDict()
        for file in result_files:
            window = get_window(file)
            data = read_alch_file(file, self.namd_ver, self.iter)
            data = np.array([data])
            if window not in all_data:
                all_data[window] = data
            else:
                #stack reps of same window together
                all_data[window] = np.vstack([all_data[window], data])

        # appending all windows together to make final array
        concat_windows = np.stack([x for x in all_data.values()], axis=0)
        #resuffle axis to be in order reps, windows, lambda_dimensions, iterations
        concat_windows = np.transpose(concat_windows, (1, 0, 2, 3))

        return concat_windows

def read_alch_file(file_path, namd_ver, iterations):
    '''

    :param file_path: str, location of namd alch file
    :param namd_ver: float, new or old used to specify what format of namd alch file we are looking at (old <= 2.12)
    :param iterations: int, Number sample in alch file
    :return: numpy array, contains potentials from one namd alch file
    '''
    # final data has order sterics appear/dis elec appear/dis to match openmm
    data = np.zeros([4, iterations])
    with open(file_path) as f:
        count = 0
        for line in f:
            if line[0:2] == 'TI':
                split_line = line.split()
                if namd_ver > 2.12:
                    data[:, count] = [float(split_line[6]), float(split_line[12]),
                                      float(split_line[4]), float(split_line[10])]
                elif namd_ver <= 2.12:
                    data[:, count] = [float(split_line[4]), float(split_line[8]),
                                      float(split_line[2]), float(split_line[6])]
                else:
                    raise ValueError('Unknown NAMD ver. {}'.format(alch))

                count += 1
    return data

def get_window(string):
    path = os.path.normpath(string)
    return float(path.split(os.sep)[-4].split('_')[1])