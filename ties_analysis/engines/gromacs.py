#!/usr/bin/env python

__copyright__ = """
    Copyright 2021 Alexander David Wade
    This file is part of TIES MD

    TIES MD is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    TIES MD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.
    You should have received a copy of the Lesser GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__license__ = "LGPL"

import os
import glob
import collections
import numpy as np

from ties_analysis.engines.openmm import Lambdas
from ties_analysis.methods.TI import TI_Analysis

class GMX_Lambdas():
    '''
    Class: holds the information about a lambda schedule in an easily queryable format (GMX)
    '''

    def __init__(self, mass, coul, vdw, bonded, restraint):
        '''

        :param mass: list of floats, describes lambda schedule for mass
        :param coul: list of floats, describes lambda schedule for columb
        :param vdw: list of floats, describes lambda schedule for vdw
        :param bonded list of floats, describes lambda schedule for bonded
        :param restraint list of floats, describes lambda schedule for restraint
        '''
        self.lambda_mass = [float(x) for x in mass]
        self.lambda_coul = [float(x) for x in coul]
        self.lambda_vdw = [float(x) for x in vdw]
        self.lambda_bonded = [float(x) for x in bonded]
        self.lambda_res = [float(x) for x in restraint]

        assert (len(self.lambda_mass) == len(self.lambda_coul))
        assert (len(self.lambda_coul) == len(self.lambda_vdw))
        assert (len(self.lambda_vdw) == len(self.lambda_bonded))
        assert (len(self.lambda_bonded) == len(self.lambda_res))

        self.schedule = []
        for i, (m, c, v, b, r) in enumerate(zip(self.lambda_mass, self.lambda_coul, self.lambda_vdw,
                                                self.lambda_bonded, self.lambda_res)):
            param_vals = {'lambda_mass': m, 'lambda_coul': c, 'lambda_vdw': v,
                          'lambda_bonded': b, 'lambda_res': r}
            self.schedule.append(param_vals)

    def update_attrs_from_schedule(self):
        '''
        helper function to update the values of self.lambda_mass etc if the self.schedule is changed
        '''
        self.lambda_mass = [x['lambda_mass'] for x in self.schedule]
        self.lambda_vdw = [x['lambda_vdw'] for x in self.schedule]
        self.lambda_coul = [x['lambda_coul'] for x in self.schedule]
        self.lambda_bonded = [x['lambda_bonded'] for x in self.schedule]
        self.lambda_res = [x['lambda_res'] for x in self.schedule]

class Gromacs(object):
    '''
    Class to perform TIES analysis on GROMACS results
    '''
    def __init__(self, method, output, win_mask, distributions, rep_convg, sampling_convg,
                 lambda_mass, lambda_coul, lambda_vdw, lambda_bonded, lambda_restraint, iterations):
        '''
        :param method: str, 'TI' or 'FEP'
        :param output: str, pointing to base dir of where output will be writen
        :param win_mask: list of ints, what windows if any to remove from analysis
        :param distributions bool, Do we want to calculate the dG for each rep individually
        :param rep_convg: list of ints, what intermediate number of reps do you wish to inspect convergence for
        :param sampling_convg: list of ints, what intermediate amount of sampling do
         you wish to inspect convergence for
        :param lambda_mass: list of floats, describes lambda schedule for mass
        :param lambda_coul: list of floats, describes lambda schedule for columb
        :param lambda_vdw: list of floats, describes lambda schedule for vdw
        :param lambda_bonded list of floats, describes lambda schedule for bonded
        :param lambda_restraint list of floats, describes lambda schedule for restraint
        :param iterations: int, number of iterations to expect in alch files
        '''

        self.name = 'Gromacs'
        self.iter = int(iterations)
        self.method = method
        self.gro_lambs = GMX_Lambdas(lambda_mass, lambda_coul, lambda_vdw, lambda_bonded, lambda_restraint)
        self.output = output
        self.win_mask = win_mask
        self.distributions = distributions
        self.rep_convg = rep_convg
        self.sampling_convg = sampling_convg

    def run_analysis(self,  data_root, temp, prot, lig, leg):
        '''

        :param data_root: str, file path point to base dir for results files
        :param temp: float for temperature in units of kelvin (not used in NAMD as FEP not implemented)
        :param prot: str, name of dir for protein
        :param lig:  str, name of dir for ligand
        :param leg: str, name of dir for thermo leg
        :return: list of floats, [dg, stdev(dg)]
        '''

        if self.method == 'FEP':
            raise NotImplemented()

        data = self.collate_data(data_root, prot, lig, leg)
        analysis_dir = os.path.join(self.output, self.name, self.method, prot, lig, leg)

        method_run = TI_Analysis(data, self.gro_lambs, analysis_dir)
        result = method_run.analysis(self.distributions, self.rep_convg, self.sampling_convg, self.win_mask)

        return result

    def collate_data(self, data_root, prot, lig, leg):
        '''
        Function to iterate over replica and window dirs reading GROMACS xvg file and building numpy array of potentials
        :param data_root: str, file path point to base dir for results files
        :param prot: str, name of dir for protein
        :param lig:  str, name of dir for ligand
        :param leg: str, name of dir for thermo leg
        :return: numpy.array for all potential collected
        '''

        results_dir_path = os.path.join(data_root, prot, lig, leg)

        result_files = os.path.join(results_dir_path, 'LAMBDA_*', 'rep*', 'simulation', 'prod.xvg')
        result_files = list(glob.iglob(result_files))

        if len(result_files) == 0:
            raise ValueError('{} in methods but no results files found'.format(self.method))

        # Sort by order of windows
        result_files.sort(key=get_window)

        print('Processing files...')
        for file in result_files:
            print(file)

        # Use ordered dict to preserve windows order
        all_data = collections.OrderedDict()
        for file in result_files:
            window = get_window(file)
            data = read_prod_file(file, self.iter)
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

def read_prod_file(file_path, iterations):
    '''

    :param file_path: str, location of GROMACS xvg file
    :param iterations: int, Number sample in GROMACS xvg file
    :return: numpy array, contains potentials from one GROMACS file
    '''
    # final data has order sterics appear/dis elec appear/dis to match openmm
    data = np.zeros([5, iterations])
    with open(file_path) as f:
        count = 0
        for line in f:
            if line[0:1] != '@' and line[0:1] != '#':
                split_line = line.split()
                #mass, coul, vdw, bonded, restraint
                data[:, count] = [float(split_line[1]), float(split_line[2]),
                                  float(split_line[3]), float(split_line[4]),
                                  float(split_line[5])]
                count += 1
    return data

def get_window(string):
    path = os.path.normpath(string)
    return float(path.split(os.sep)[-4].split('_')[1])
