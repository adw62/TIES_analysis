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

import copy
import numpy as np
from sklearn.utils import resample

class TI_Analysis(object):
    '''
    Class for thermodynamic integration analysis
    '''

    def __init__(self, grads, lambdas, analysis_dir):
        '''

        :param grads: numpy array for all results
        :param lambdas: Lambda class containing schedule
        :param distribution: Boolean, if True then dGs will not be averaged and a distribution of results is returned
        :param analysis_dir: string, pointing to where we want analysis saved
        '''

        self.data = grads

        self.lambdas = lambdas
        self.shape = list(self.data.shape)
        self.nstates = self.shape[1]

        #directory where any TI specific output could be saved, not curretly used
        self.analysis_dir = analysis_dir

        print('Data shape is {} repeats, {} states, {} lambda dimensions, {} iterations\n'.format(*self.shape))

    def analysis(self, distributions=False, rep_convg=None, sampling_convg=None, mask_windows=None):
        '''
        Perform TI analysis
        :param distributions bool, Do we want to calculate the dG for each rep individually
        :param mask_windows list of ints, specify windows to remove from analysis
        :return: list, dG calculated by TI and associated standard deviation
        '''

        data = copy.deepcopy(self.data)
        lambdas = copy.deepcopy(self.lambdas)

        # remove windows if requested
        if mask_windows is not None:
            mask_windows.sort()
            mask_windows.reverse()
            print('Removing windows {}'.format(mask_windows))
            for win in mask_windows:
                data = np.delete(data, win, axis=1)
                del lambdas.schedule[win]
        lambdas.update_attrs_from_schedule()

        #iterate over sampling and compute dg if requested
        if sampling_convg is not None:
            sampling_free_energy = []
            for sampling in sampling_convg:
                avg_over_iterations = np.average(data[:, :, :, 0:sampling], axis=3)
                free_energy, variance = self.intergrate(avg_over_iterations.transpose(), lambdas)
                result = [sum(free_energy.values()), np.sqrt(sum(variance.values()))]
                sampling_free_energy.append(result)
            print('Convergence with number of samples:')
            print(sampling_convg)
            print(sampling_free_energy)
            print('')

        avg_over_iterations = np.average(data, axis=3)
        #compute dgs seperatly for each replica if requested
        if distributions:
            dist_free_energy = []
            for rep in avg_over_iterations:
                free_energy, _ = self.intergrate(np.array([rep]).transpose(), lambdas)
                dist_free_energy.append(sum(free_energy.values()))
            print('dG for each replica:')
            print(dist_free_energy)
            print('')

        #iterate overs chuncks of replicas and compute dg if requested
        if rep_convg is not None:
            rep_free_energy = []
            for rep in rep_convg:
                free_energy, variance = self.intergrate(avg_over_iterations[0:rep].transpose(), lambdas)
                result = [sum(free_energy.values()), np.sqrt(sum(variance.values()))]
                rep_free_energy.append(result)
            print('Convergence with number of reps:')
            print(rep_convg)
            print(rep_free_energy)
            print('')

        #compute dg for all reps and sampling this is the result that is returned
        free_energy, variance = self.intergrate(avg_over_iterations.transpose(), lambdas)

        print('Free energy breakdown:')
        print(free_energy)
        print('Variance breakdown:')
        print(variance)
        print('')

        return [sum(free_energy.values()), np.sqrt(sum(variance.values()))]

    def intergrate(self, data, lambdas):
        '''
        :param data: numpy array of averaged gradients
        :param lambdas: class, contains information about lambda schedule
        :return: turple of dicts, {lambda_parameters: free energy}, {lambda_parameters: free energy variance} for
         each lambda parameter
        '''

        lambda_names = lambdas.schedule[0].keys()
        free_energy = {k: 0.0 for k in lambda_names}
        var = {k: 0.0 for k in lambda_names}

        # iterate over lambda dimensions
        for i, lam_name in enumerate(lambda_names):
            lambda_array = getattr(lambdas, lam_name)

            dlam = get_lam_diff(lambda_array)

            # if dlam is not zero then bootstrap the replicas to get avg and var
            avg_var_by_state = np.array([compute_bs_error(state_data) if x != 0.0 else [0.0, 0.0] for
                                         state_data, x in zip(data[i], dlam)])

            avg_by_state = np.array([x[0] for x in avg_var_by_state])
            var_by_state = np.array([x[1] for x in avg_var_by_state])

            free_energy[lam_name] = np.trapz(avg_by_state, lambda_array)
            var[lam_name] = np.sum(np.square(dlam) * var_by_state)

        return free_energy, var

def get_lam_diff(lambda_array):
    '''

    :param lambda_array: list of ints
    :return: numpy array for differences in input
    '''
    lambda_mp = np.array(np.multiply(0.5, np.add(lambda_array[1:], lambda_array[:-1])))
    lambda_mp = np.insert(lambda_mp, 0, lambda_array[0])
    lambda_mp = np.append(lambda_mp, lambda_array[-1])

    dlam = np.array(np.subtract(lambda_mp[1:], lambda_mp[:-1]))
    return dlam


def compute_bs_error(replicas):
    '''
    compute bootstrapped average and variance
    :param replicas, list, values of average dU/dlam in each replica
    :return: turple of floats, average and var of boot strapped result
    '''
    if len(replicas) > 1:
        boot = [resample(replicas, replace=True, n_samples=len(replicas)) for x in range(5000)]
        avg_per_boot_sample = np.average(boot, axis=1)
    else:
        print('Warning asked for variance of 1 replica')
        return replicas[0], 0.0

    return np.average(avg_per_boot_sample), np.var(avg_per_boot_sample, ddof=1)



