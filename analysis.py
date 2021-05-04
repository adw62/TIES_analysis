#!/usr/bin/env python

import glob
import os


import numpy as np
import scipy.stats
from pymbar import MBAR, timeseries
import os
import copy
from sklearn.utils import resample
import collections

from arch.bootstrap import IIDBootstrap
import json

skip_graphs = False
try:
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import colors

    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=2.5)

except ModuleNotFoundError:
    print('seaborn and or matplotlib not found skipping graphs')
    skip_graphs = True


class MBAR_Analysis():
    '''
    Class for MBAR analysis
    '''
    def __init__(self, MBARs, lambdas, analysis_dir):
        '''

        :param MBARs: numpy array for all results
        :param lambdas: Lambda class containing schedule
        :param analysis_dir, string
        '''

        #HARD CODING FOR PAPER ANALYSIS
        self.data = MBARs[0:5, :, :, :]

        self.shape = list(self.data.shape)
        self.nstates = self.shape[1]
        print('Data shape is {} repeats, {} states_i, {} states_j, {} iterations'.format(*self.shape))

        self.lambdas = lambdas
        self.analysis_dir = analysis_dir

        self.data, self.N_k = MBAR_Analysis.decorrelate_data(self)

    def decorrelate_data(self):
        '''

        decorrolate time series data.
        :return turple, (np matrix containing decorrelated data, list of ints for end of decorrelated data in matrix)
        '''

        decorr_data_sav = os.path.join(self.analysis_dir, 'fep_decorr_data.npy')
        index_sav = os.path.join(self.analysis_dir, 'fep_N_k.npy')

        rep_decorr_data_sav = os.path.join(self.analysis_dir, 'fep_rep_decorr_data.npy')
        rep_index_sav = os.path.join(self.analysis_dir, 'fep_rep_N_k.npy')

        '''
        if os.path.exists(decorr_data_sav) and os.path.exists(index_sav):
            decorr_data = np.load(decorr_data_sav)
            N_k = np.load(index_sav)
            return decorr_data, N_k
        '''

        print('Decorrelating data...')
        iter_per_rep = self.shape[3]
        num_repeats = self.shape[0]
        tot_iter = iter_per_rep*num_repeats

        #decorr data matrix is square for numpy convenience and padded with zeros.
        # N_k allows know were the data ends and the padding begins in the matrix.
        N_k = np.zeros([self.nstates], np.int32)
        decorr_data = np.zeros([self.nstates, self.nstates, tot_iter])

        #to compute ensemble stdev we also would like to save the result for each replica speratly
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
                decorr_data[k, :, 0+N_k[k]:N_k[k]+len(sub_idx)] = u_kln[k, :, sub_idx].T
                rep_deccor_data[k, :, 0:len(sub_idx)] = u_kln[k, :, sub_idx].T
                N_k[k] += len(sub_idx)
                rep_Nk[k] = len(sub_idx)
            replicas_deccor.append(rep_deccor_data)
            replicas_Nk.append(rep_Nk)

        print('Number of decorrelated samples extracted per state: {}'.format(N_k))

        np.save(decorr_data_sav, decorr_data)
        np.save(index_sav, N_k)

        np.save(rep_decorr_data_sav, replicas_deccor)
        np.save(rep_index_sav, replicas_Nk)

        return decorr_data, N_k
    
    def replica_analysis(self):
        '''
        Function to make analysis of result from MBAR considering each trajectory a one replica
        :return: list, containing average of bootstrapped dG and SEM
        '''

        rep_decorr_data_sav = os.path.join(self.analysis_dir, 'fep_rep_decorr_data.npy')
        rep_index_sav = os.path.join(self.analysis_dir, 'fep_rep_N_k.npy')

        data = np.load(rep_decorr_data_sav)
        Nk = np.load(rep_index_sav)

        replica_results = []
        for d, n in zip(data, Nk):
            # truncate used as a dummy here to stop graphs being plotted
            results = self.analysis(u_kln=d, N_k=n, truncate=2000)
            replica_results.append(results)

        #x[1] here is MBAR error which is not used
        dg = [x[0] for x in replica_results]
        print('dG for each replica')
        print(replica_results)

        result = compute_bs_error(dg)
        return [result[0], np.sqrt(result[1])]

        #return [np.average(dg), np.std(dg, ddof=1)/np.sqrt(len(dg))]

    def analysis(self, u_kln=None, N_k=None, truncate=None, mask_windows=None):
        '''
        Process a matrix of potentials passes to this function as u_kln or process self.data which
         is matrix of all replicas
        :param u_kln: numpy array matrix of potentials for one replica
        :param N_k: list of ints, specifies which entry in u_kln should be read up to for each window
        :param truncate: int, can be used to specify if samples are truncated, useful for convergence plots
        :param mask_windows: list of ints, can be used to specify what windows to remove from u_kln
        :return: list, containing dG and associated error calculated by MBAR
        '''

        if u_kln is None:
            u_kln = self.data
        if N_k is None:
            N_k = self.N_k

        if truncate is not None:
            u_kln = u_kln[:, :, 0:truncate]
            N_k = [truncate if x > truncate else x for x in N_k]

        #remove windows
        if mask_windows is not None:
            mask_windows.sort()
            mask_windows.reverse()
            if 0 in mask_windows or self.nstates-1 in mask_windows:
                raise ValueError('Can not remove end states')
            for win in mask_windows:
                u_kln = np.delete(u_kln, win, axis=0)
                u_kln = np.delete(u_kln, win, axis=1)
                N_k = np.delete(N_k, win, axis=0)

        # Compute free energy differences and statistical uncertainties
        mbar = MBAR(u_kln, N_k)
        [DeltaF_ij, dDeltaF_ij, _] = mbar.getFreeEnergyDifferences()

        # print("Number of uncorrelated samples per state: {}".format(N_k))
        result = ([DeltaF_ij[0, len(N_k)-1], dDeltaF_ij[0, len(N_k)-1]])

        #Save data to analysis dir

        if truncate is None and mask_windows is None:
            np.save(os.path.join(self.analysis_dir, 'DeltaF_ij.npy'), DeltaF_ij)
            np.save(os.path.join(self.analysis_dir, 'dDeltaF_ij.npy'), dDeltaF_ij)

        return result


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

        #HARD CODING FOR PAPER ANALYSIS
        self.data = grads[0:5, :, :, :]

        self.lambdas = lambdas
        self.shape = list(self.data.shape)
        self.nstates = self.shape[1]

        self.analysis_dir = analysis_dir

        #Do a check that this is the shape of data we would expect from the config file
        print('Data shape is {} repeats, {} states, {} lambda dimensions, {} iterations'.format(*self.shape))

    def analysis(self, truncate=None, mask_windows=None, rep_trunc=None):
        '''
        Perform analysis
        :param truncate: int, determines max frame to be analysed, useful for convergence plots
        :param mask_windows list of ints, specify windows to remove from analysis
        :param rep_trunc int, determines max number of replicas to be analysed
        :return: list, dG calculated by TI and associated standard deviation
        '''

        #remove sampling
        if truncate is not None:
            avg_over_iterations = np.average(self.data[:, :, :, 0:truncate], axis=3)

        #remove replicas
        if rep_trunc is not None:
            avg_over_iterations = np.average(self.data[0:rep_trunc, :, :, :], axis=3)

        #use all data
        if truncate is None and rep_trunc is None:
            avg_over_iterations = np.average(self.data, axis=3)

        lambdas = copy.deepcopy(self.lambdas)
        #remove windows
        if mask_windows is not None:
            mask_windows.sort()
            mask_windows.reverse()
            if 0 in mask_windows or self.nstates-1 in mask_windows:
                raise ValueError('Can not remove end states')
            for win in mask_windows:
                avg_over_iterations = np.delete(avg_over_iterations, win, axis=1)
                del lambdas.schedule[win]
        lambdas.update_attrs_from_schedule()

        free_energy, variance = self.intergrate(avg_over_iterations.transpose(), lambdas)

        #if truncate is None and mask_windows is None and rep_trunc is None:
        #    print('Free energy breakdown:')
        #    print(free_energy)
        #    print('Variance breakdown:')
        #    print(variance)

        return [sum(free_energy.values()), np.sqrt(sum(variance.values()))]

    def intergrate(self, data, lambdas):
        '''

        :param data: numpy array of averaged gradients
        :param lambdas: class, contains information about lambda schedual
        :return: turple of dicts, {lambda_parameters: free energy}, {lambda_parameters: free energy variance} for
         each lambda parameter
        '''

        lambda_names = lambdas.schedule[0].keys()
        free_energy = {k: 0.0 for k in lambda_names}
        var = {k: 0.0 for k in lambda_names}

        # iterate over lambda dimensions
        for i, lam_name in enumerate(lambda_names):
            lambda_array = getattr(lambdas, lam_name)

            dlam = TI_Analysis.get_lam_diff(self, lambda_array)

            #iteration over state
            #if dlam is not zero then bootstrap the repicas to get avg and var
            avg_var_by_state = np.array([compute_bs_error(state_data) if x != 0.0 else [0.0, 0.0] for
                                         state_data, x in zip(data[i], dlam)])
            avg_by_state = np.array([x[0] for x in avg_var_by_state])
            var_by_state = np.array([x[1] for x in avg_var_by_state])


            #avg_by_state = [x if y != 0.0 else 0.0 for x, y in zip(avg_by_state, dlam)]
            #var_by_state = [x if y != 0.0 else 0.0 for x, y in zip(var_by_state, dlam)]

            free_energy[lam_name] = np.trapz(avg_by_state, lambda_array)
            #xx = np.sum(np.multiply(avg_by_state, dlam))
            #print(free_energy[lam_name], xx)
            var[lam_name] = np.sum(np.square(dlam)*var_by_state)

        return free_energy, var

    def get_lam_diff(self, lambda_array):
        '''
        Note np.diff could be used instead
        :param lambda_array: list of ints
        :return: numpy array for differences in input
        '''
        lambda_mp = np.array(np.multiply(0.5, np.add(lambda_array[1:], lambda_array[:-1])))
        lambda_mp = np.insert(lambda_mp, 0, lambda_array[0])
        lambda_mp = np.append(lambda_mp, lambda_array[-1])

        dlam = np.array(np.subtract(lambda_mp[1:],lambda_mp[:-1]))
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

class Lambdas():
    '''
    Class: holds the information about a lambda schedual in an easily queryable format
    '''
    def __init__(self, vdw_a, vdw_d, ele_a, ele_d):

        self.lambda_sterics_appear = [float(x) for x in vdw_a]
        self.lambda_electrostatics_appear =[float(x) for x in ele_a]
        self.lambda_sterics_disappear = [float(x) for x in vdw_d]
        self.lambda_electrostatics_disappear = [float(x) for x in ele_d]

        assert (len(self.lambda_sterics_appear) == len(self.lambda_electrostatics_appear))
        assert (len(self.lambda_sterics_disappear) == len(self.lambda_electrostatics_appear))
        assert (len(self.lambda_sterics_disappear) == len(self.lambda_electrostatics_disappear))

        self.schedule = []
        for i, (su, sd, eu, ed) in enumerate(zip(self.lambda_sterics_appear, self.lambda_sterics_disappear,
                                                 self.lambda_electrostatics_appear,
                                                 self.lambda_electrostatics_disappear)):
            param_vals = {'lambda_sterics_appear': su, 'lambda_sterics_disappear': sd,
                          'lambda_electrostatics_appear': eu, 'lambda_electrostatics_disappear': ed}
            self.schedule.append(param_vals)

    def update_attrs_from_schedule(self):
        '''
        helper function to update the values of self.lambda_sterics_appear etc if the self.schedule is changed
        :return:
        '''
        self.lambda_sterics_appear = [x['lambda_sterics_appear'] for x in self.schedule]
        self.lambda_electrostatics_appear = [x['lambda_electrostatics_appear'] for x in self.schedule]
        self.lambda_sterics_disappear = [x['lambda_sterics_disappear'] for x in self.schedule]
        self.lambda_electrostatics_disappear = [x['lambda_electrostatics_disappear'] for x in self.schedule]

def nice_print(string):
    string = string.center(75, '#')
    print(string)

def read_config(config_file):
    '''

    :param config_file: str, path to a config file
    :return: dict, containg all options to run analysis
    '''
    args_dict = {}
    with open(config_file) as file:
        for line in file:
            if line[0] != '#' and line[0] != ' ' and line[0] != '\n':
                data = line.rstrip('\n').split('=')
                if len(data) > 2:
                    raise ValueError('Failed to parse line: {}'.format(line))
                # Remove spaces
                data = [s.replace(" ", "") for s in data]
                data = [s.replace("\t", "") for s in data]
                args_dict[data[0]] = data[1]
    args_dict = {k: v.split(',') for k, v in args_dict.items()}

    return args_dict


def collate_openmm_data(results_dir, method):
    '''

    :param results_dir: str, data dir pointing to base for the results of a specific ligand and thermo leg
    :return: numpy array, contains potentials sampled from all alchemical windows and repeats
    '''

    if method == 'TI':
        TI_results_files = os.path.join(results_dir, 'TI*.npy')
        result_files = list(glob.iglob(TI_results_files))
    else:
        FEP_results_files = os.path.join(results_dir, 'FEP*.npy')
        result_files = list(glob.iglob(FEP_results_files))

    if len(result_files) == 0:
        raise ValueError('{} in methods but no results files found'.format(method))

    # Sort by order of windows
    result_files.sort(key=sort_results)

    #print('Processing files...')
    #for file in result_files:
    #    print(file)

    # Use ordered dict to preserve windows order
    data = collections.OrderedDict()
    for file in result_files:
        window = file.split('/')[-1].split('_')[1]
        loaded = np.load(file, allow_pickle=True)
        loaded = np.array([loaded])
        if window not in data:
            data[window] = loaded
        else:
            # appending all repeats to a new axis
            data[window] = np.vstack([data[window], loaded])

    # appending all windows together to make final array
    concat_windows = np.array(np.concatenate([v for v in data.values()], axis=1))

    return concat_windows


def sort_results(string):
    '''
    Helper function to sort directory paths by specific index in file name
    :param string:
    :return:
    '''
    return int(string.split('/')[-1].split('_')[1])


def collate_namd_data(results_dir, alch, iterations=2001):
    '''

    :param results_dir: str, data dir pointing to base for the results of a specific ligand and thermo leg
    :param alch: str, new or old used to specify what format of namd alch file we are looking at (old <= 2.12)
    :param iterations: int, Number sample in alch file
    :return: numpy array, contains potentials sampled from all alchemical windows and repeats
    '''
    alch_wc = os.path.join(results_dir, 'LAMBDA_*', 'rep*', 'simulation', 'sim1.alch')
    alch_filepaths = list(glob.iglob(alch_wc))
    all_data={}
    for file_path in alch_filepaths:
        lamb_idx = float(file_path.split('LAMBDA_')[1][0:4])
        #rep_idx  = int(file_path.split('rep')[1][0])
        data = read_alch_file(file_path, alch, iterations)
        data = np.array([data])
        if lamb_idx not in all_data:
            all_data[lamb_idx] = data
        else:
            all_data[lamb_idx] = np.vstack([all_data[lamb_idx], data])

    ordered_data = collections.OrderedDict()

    for x in sorted(all_data.keys()):
        ordered_data[x] = all_data[x]
    del all_data

    # appending all windows together to make final array
    concat_windows = np.stack([x for x in ordered_data.values()], axis=0)
    concat_windows = np.transpose(concat_windows, (1, 0, 2, 3))

    return concat_windows

def read_alch_file(file_path, alch, iterations):
    '''

    :param file_path: str, location of namd alch file
    :param alch: str, new or old used to specify what format of namd alch file we are looking at (old <= 2.12)
    :param iterations: int, Number sample in alch file
    :return: numpy array, contains potentials from one namd alch file
    '''
    #final data has order sterics appear/dis elec appear/dis to match openmm
    data = np.zeros([4, iterations])
    with open(file_path) as f:
        count = 0
        for line in f:
            if line[0:2] == 'TI':
                split_line = line.split()
                if alch == 'new':
                    data[:, count] = [float(split_line[6]), float(split_line[12]),
                              float(split_line[4]), float(split_line[10])]
                elif alch == 'old':
                    data[:, count] = [float(split_line[4]), float(split_line[8]),
                                  float(split_line[2]), float(split_line[6])]
                else:
                    raise ValueError('Unknown NAMD ver. {}'.format(alch))

                count += 1
    return data


def r2(x, y):
    slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(x, y)
    return r_value ** 2


def slope_fun(x, y):
    slope, intercept, r_value, p_value, stderr = scipy.stats.linregress(x, y)
    return slope


def mse(x, y):
    return np.square(np.subtract(x, y)).mean()


def mue(x, y):
    error = np.abs(np.array(x) - np.array(y))
    return error.mean()


def rmsd(x, y):
    deviation = np.array(x) - np.array(y)
    return np.sqrt((deviation ** 2).mean())


def spearman(x, y):
    correlation, p_value = scipy.stats.spearmanr(x, y)
    return correlation


def person(x, y):
    correlation, p_value = scipy.stats.pearsonr(x, y)
    return correlation


def kendall_tau(x, y):
    correlation, p_value = scipy.stats.kendalltau(x, y)
    return correlation


def get_result(engine, method, sim_file_path, namd_lambs, openmm_lambs):
    if method == 'TI':

        if engine == 'N3':
            data = collate_namd_data(sim_file_path, alch='new')
            ana = TI_Analysis(data, namd_lambs, './test')

        elif engine == 'N2':
            data = collate_namd_data(sim_file_path, alch='old')
            ana = TI_Analysis(data, namd_lambs, './test')

        elif engine == 'OpenMM':
            data = collate_openmm_data(sim_file_path, method)
            ana = TI_Analysis(data, openmm_lambs, './test')

        else:
            raise ValueError('Unknown engine {}'.format(engine))

        return ana.analysis()

    elif method == 'FEP':

        #NAMD FEP data can be collected with OpenMM functions
        if engine == 'N3':
            data = collate_openmm_data(sim_file_path, method)
            ana = MBAR_Analysis(data, namd_lambs, './test')

        elif engine == 'N2':
            data = collate_openmm_data(sim_file_path, method)
            ana = MBAR_Analysis(data, namd_lambs, './test')

        elif engine == 'OpenMM':
            data = collate_openmm_data(sim_file_path, method)
            ana = MBAR_Analysis(data, openmm_lambs, './test')

        else:
            raise ValueError('Unknown engine {}'.format(engine))

        return ana.replica_analysis()

    else:
        raise ValueError('Unknown method {}'.format(method))


def compute_stats(x_dict, y_dict, master_keys):
    '''

    :param x_dict: dict, keyed by str of ligand transformation name with value for its ddG
     for some method (example method:NAMD2-TI)
    :param y_dict: dict, keyed by str of ligand transformation name with value for its ddG for some other method
    :param master_keys: list of str, str index the ligand transfomations
    :return: list of lists: contains value calculated for each stats function and associated CI at 95%
    '''

    x = np.array([x_dict[k][0] for k in master_keys])
    y = np.array([y_dict[k][0] for k in master_keys])

    stats_functions = [mue, mse, rmsd, person, slope_fun, r2]
    bs = IIDBootstrap(x, y)

    stats_estimates = [func(x, y) for func in stats_functions]

    stats_cis = [bs.conf_int(func, reps=5000, method='basic', size=0.95) for func in stats_functions]

    for i, (val, err) in enumerate(zip(stats_estimates, stats_cis)):
        if i in [0, 1, 2]:
            print(val, val - err[0][0])
        else:
            print(val)

def process_results(full_results_dict, exp_data):

    for prot_name, prot_data in exp_data.items():
        n3_ti = full_results_dict['N3']['TI'][prot_name]
        n3_fep = full_results_dict['N3']['FEP'][prot_name]

        n2_ti = full_results_dict['N2']['TI'][prot_name]
        n2_fep = full_results_dict['N2']['FEP'][prot_name]

        openmm_ti = full_results_dict['OpenMM']['TI'][prot_name]
        openmm_fep = full_results_dict['OpenMM']['FEP'][prot_name]

        all_data = [n3_ti, n3_fep, n2_ti, n2_fep, openmm_ti, openmm_fep]

        for x in range(0, 6):
            compute_stats(all_data[x], exp_data[prot_name], exp_data[prot_name].keys())

if __name__ == '__main__':
    nice_print('Analysis')

    #NAMD args to set lambda windows
    namd_args = read_config('./namd.cfg')
    namd_lambs = Lambdas(namd_args['vdw_a'], namd_args['vdw_d'], namd_args['ele_a'], namd_args['ele_d'])

    #OpenMM args to set lambda windows
    openmm_args = read_config('./openmm.cfg')
    openmm_lambs = Lambdas(openmm_args['vdw_a'], openmm_args['vdw_d'], openmm_args['ele_a'], openmm_args['ele_d'])

    #genral args specify where data is located what methods to use
    genral_args = read_config('./analysis.cfg')

    #EXP data specifies what protiens and ligands to look at
    exp_data = eval(open(genral_args['exp_data'][0]).read())

    engines = ['N3', 'N2', 'OpenMM']

    if os.path.exists('./all_res.dat'):
        with open('file.txt', 'w') as f:
            result = eval(open('./all_res.dat').read())

    else:
        result = {}

        #iterate over molecular dynamics engiens
        for engine in engines:
            nice_print(engine)

            result[engine] = {}

            #iterate over alchemical methods
            for method in genral_args['methods']:
                nice_print(method)

                result[engine][method] = {}

                #OpenMM and NAMD TIES store there results in differnt locations
                if engine == 'OpenMM':
                    simulation_legs = ['complex', 'solvent']
                    file_suffix = 'results/complex'
                    data_dir = os.path.join(genral_args['data_root'][0], engine, '*')
                else:
                    if method == 'TI':
                        file_suffix = ''
                        simulation_legs = ['com', 'lig']
                    else:
                        file_suffix = 'results/complex'
                        if engine == 'N2':
                            simulation_legs = ['complex', 'solvent']
                        else:
                            simulation_legs = ['com', 'lig']
                    data_dir = os.path.join(genral_args['data_root'][0], engine, method, '*')

                prot_filepaths = list(glob.iglob(data_dir))

                #iterate over protiens
                for prot_name, prot_data  in exp_data.items():
                    nice_print(prot_name)

                    result[engine][method][prot_name] = {}

                    #Check we can find this protein in the data dir
                    user_selected_prot = [x for x in prot_filepaths if prot_name == os.path.split(x)[-1]]
                    if len(user_selected_prot) == 0:
                        raise ValueError('Cannot find data dirs for {}'.format(prot_name))
                    user_selected_prot = user_selected_prot[0]

                    ligand_filepaths = list(glob.iglob(os.path.join(user_selected_prot, 'l*l*')))

                    # iterate over ligands
                    for lig_name, lig_data in prot_data.items():
                        nice_print(lig_name)

                        result[engine][method][prot_name][lig_name] = {}

                        #Check we can find the ligand in data dir
                        user_selected_ligs = [x for x in ligand_filepaths if lig_name == os.path.split(x)[-1]]
                        if len(user_selected_ligs) == 0:
                            continue
                        else:
                            user_selected_ligs = user_selected_ligs[0]

                        #iterate over legs of the simulation ie: solvation
                        res = {'lig': 0.00, 'com': 0.00}
                        for sim in simulation_legs:
                            sim_file_path = os.path.join(user_selected_ligs, sim, file_suffix)

                            # renaming of legs to make NAMD and OpenMM the have same names.
                            if sim == 'complex':
                                sim_name = 'com'
                            elif sim == 'solvent':
                                sim_name = 'lig'
                            else:
                                sim_name = sim
                            nice_print(sim_name)

                            res[sim_name] = get_result(engine, method, sim_file_path, namd_lambs, openmm_lambs)
                            print('dG = {}: SEM = {}'.format(res[sim_name][0], res[sim_name][1]))

                        #Caution here as exp result maybe calculated as -1* this
                        ddG = res['lig'][0] - res['com'][0]
                        ddG_err = np.sqrt(np.square(res['lig'][1]) + np.square(res['com'][1]))
                        result[engine][method][prot_name][lig_name] = [ddG, ddG_err]

        with open('./all_res.dat', 'w') as f:
            f.write(json.dumps(result))

    process_results(result, exp_data)



