#!/usr/bin/env python

from ties_analysis.engines.openmm import OpenMM
from ties_analysis.engines.namd import NAMD

class Config():
    def __init__(self, namd_cfg='./namd.cfg', openmm_cfg='./openmm.cfg', analysis_cfg='./analysis.cfg'):
        '''

        :param namd_cfg: str, path to file containing namd analysis params
        :param openmm_cfg: path to file containing openmm analysis params
        :param analysis_cfg: path to file containing general analysis params
        '''

        # process general arg for analysis
        # general args specify where data is located what methods to use and any other parameters
        general_args = read_config(analysis_cfg)

        self.engines_to_init = general_args['engines']
        self.simulation_legs = general_args['legs']
        self.data_root = general_args['data_root'][0]
        self.analysis_dir = general_args['output_dir'][0]

        # what alchemical windows to remove from analysis
        if general_args['windows_mask'][0] != 'None':
            self.win_mask = [int(x) for x in general_args['windows_mask']]
        else:
            self.win_mask = None

        # Process engines
        self.engines = []
        if 'NAMD2' in self.engines_to_init or 'NAMD3' in self.engines_to_init:
            namd_args = read_config(namd_cfg)
            namd_args['namd_version'] = namd_args['namd_version'][0]
            namd_args['iterations'] = namd_args['iterations'][0]
            methods = namd_args['methods']
            del namd_args['methods']
            for method in methods:
                namd = NAMD(method, self.analysis_dir, self.win_mask, **namd_args)
                self.engines.append(namd)

        if 'OpenMM' in self.engines_to_init:
            openmm_args = read_config(openmm_cfg)
            methods = openmm_args['methods']
            del openmm_args['methods']
            for method in methods:
                openmm = OpenMM(method, self.analysis_dir, self.win_mask, **openmm_args)
                self.engines.append(openmm)

        if len(self.engines) == 0:
            raise ValueError('No support engines requested. Please choose from (NAMD2/NAMD3/OpenMM)')

        #process exp data
        # EXP data specifies what proteins and ligands to look at
        self.exp_data = eval(open(general_args['exp_data'][0]).read())


def read_config(config_file):
    '''

    :param config_file: str, path to a config file
    :return: dict, containing all options to run analysis
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