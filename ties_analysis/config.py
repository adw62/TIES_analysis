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

from ties_analysis.engines.openmm import OpenMM
from ties_analysis.engines.namd import NAMD
from ties_analysis.engines.gromacs import Gromacs

class Config():
    def __init__(self, analysis_cfg='./analysis.cfg'):
        '''

        :param analysis_cfg: path to file containing general analysis params
        '''
        namd_cfg = './namd.cfg'
        openmm_cfg = './openmm.cfg'
        gromacs_cfg = './gromacs.cfg'
        # process general arg for analysis
        # general args specify where data is located what methods to use and any other parameters
        general_args = read_config(analysis_cfg)

        self.temp = float(general_args['temperature'][0]) #unit = kelvin
        self.distributions = bool(int(general_args['distributions'][0]))
        self.engines_to_init = [x.lower() for x in general_args['engines']]
        self.simulation_legs = general_args['legs']
        self.data_root = general_args['data_root'][0]
        self.analysis_dir = general_args['output_dir'][0]
        self.rep_convg = general_args['rep_convg'][0]
        self.sampling_convg = general_args['sampling_convg'][0]

        # what alchemical windows to remove from analysis
        if general_args['windows_mask'][0] != 'None':
            self.win_mask = [int(x) for x in general_args['windows_mask']]
        else:
            self.win_mask = None

        if general_args['rep_convg'][0] != 'None':
            self.rep_convg = [int(x) for x in general_args['rep_convg']]
        else:
            self.rep_convg = None

        if general_args['sampling_convg'][0] != 'None':
            self.sampling_convg = [int(x) for x in general_args['sampling_convg']]
        else:
            self.sampling_convg = None

        # Process engines
        self.engines = []
        if 'namd2' in self.engines_to_init or 'namd3' in self.engines_to_init:
            namd_args = read_config(namd_cfg)
            namd_args['namd_version'] = namd_args['namd_version'][0]
            methods = general_args['methods']
            for method in methods:
                namd = NAMD(method, self.analysis_dir, self.win_mask, self.distributions, self.rep_convg,
                            self.sampling_convg, **namd_args)
                self.engines.append(namd)

        if 'openmm' in self.engines_to_init:
            openmm_args = read_config(openmm_cfg)
            methods = general_args['methods']
            for method in methods:
                openmm = OpenMM(method, self.analysis_dir, self.win_mask, self.distributions, self.rep_convg,
                                self.sampling_convg, **openmm_args)
                self.engines.append(openmm)

        if 'gromacs' in self.engines_to_init:
            gro_args = read_config(gromacs_cfg)
            methods = general_args['methods']
            gro_args['iterations'] = gro_args['iterations'][0]
            for method in methods:
                gro = Gromacs(method, self.analysis_dir, self.win_mask, self.distributions, self.rep_convg,
                              self.sampling_convg, **gro_args)
                self.engines.append(gro)

        if len(self.engines) == 0:
            raise ValueError('No support engines requested. Please choose from (NAMD2/NAMD3/OpenMM/GROMACS)')

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