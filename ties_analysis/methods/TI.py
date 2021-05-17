#!/usr/bin/env python

import copy
import numpy as np
from sklearn.utils import resample
from pathlib import Path

class TI_Analysis(object):
    '''
    Class for thermodynamic integration analysis
    '''

    def __init__(self, grads, lambdas, analysis_dir):
        '''

        :param grads: numpy array for all results
        :param lambdas: Lambda class containing schedule
        :param analysis_dir: string, pointing to where we want analysis saved
        '''

        self.data = grads

        self.lambdas = lambdas
        self.shape = list(self.data.shape)
        self.nstates = self.shape[1]

        self.analysis_dir = analysis_dir
        print('Attempting to make analysis folder {0}'.format(self.analysis_dir))
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)

        # Do a check that this is the shape of data we would expect from the config file
        print('Data shape is {} repeats, {} states, {} lambda dimensions, {} iterations'.format(*self.shape))

    def analysis(self, mask_windows=None):
        '''
        Perform TI analysis
        :param mask_windows list of ints, specify windows to remove from analysis
        :return: list, dG calculated by TI and associated standard deviation
        '''

        avg_over_iterations = np.average(self.data, axis=3)

        lambdas = copy.deepcopy(self.lambdas)
        # remove windows
        if mask_windows is not None:
            mask_windows.sort()
            mask_windows.reverse()
            print('Removing windows {}'.format(mask_windows))
            for win in mask_windows:
                avg_over_iterations = np.delete(avg_over_iterations, win, axis=1)
                del lambdas.schedule[win]
        lambdas.update_attrs_from_schedule()

        free_energy, variance = self.intergrate(avg_over_iterations.transpose(), lambdas)

        print('Free energy breakdown:')
        print(free_energy)
        print('Variance breakdown:')
        print(variance)

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

            # if dlam is not zero then bootstrap the repicas to get avg and var
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
    boot = [resample(replicas, replace=True, n_samples=len(replicas)) for x in range(5000)]
    avg_per_boot_sample = np.average(boot, axis=1)

    return np.average(avg_per_boot_sample), np.var(avg_per_boot_sample, ddof=1)



