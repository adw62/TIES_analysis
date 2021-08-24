#!/usr/bin/env python

import os

from .config import Config
import numpy as np

class Analysis():
    def __init__(self):

        cfg = Config()
        result = {}

        for engine in cfg.engines:
            engine_id = engine.name+'_'+engine.method
            if engine_id not in result:
                result[engine_id] = {}
            else:
                raise ValueError('Repeated engine name {}'.format(engine_id))

            for prot_name, prot_data in cfg.exp_data.items():
                if prot_name not in result[engine_id]:
                    result[engine_id][prot_name] = {}
                else:
                    raise ValueError('Repeated protein name {}'.format(prot_name))

                for lig_name, lig_data in prot_data.items():
                    if lig_name not in result[engine_id][prot_name]:
                        result[engine_id][prot_name][lig_name] = {}
                    else:
                        raise ValueError('Repeated ligand name {}'.format(lig_name))

                    nice_print('{} {} {}'.format(engine_id, prot_name, lig_name))
                    leg_results = {leg: 0.00 for leg in cfg.simulation_legs}

                    for leg in cfg.simulation_legs:
                        nice_print(leg)
                        leg_results[leg] = engine.run_analysis(cfg.data_root, cfg.temp, prot_name, lig_name, leg)

                    if len(cfg.simulation_legs) == 2:
                        print('Two thermodynamic legs found assuming this is a ddG calculation')
                        leg1 = cfg.simulation_legs[0]
                        leg2 = cfg.simulation_legs[1]
                        print('Computing {} - {}'.format(leg1, leg2))

                        ddg = leg_results[leg1][0] - leg_results[leg2][0]
                        ddg_err = np.sqrt(np.square(leg_results[leg1][1]) + np.square(leg_results[leg2][1]))
                        print('ddG = {}: SEM = {}'.format(ddg, ddg_err))
                        result[engine_id][prot_name][lig_name] = [ddg, ddg_err]

                    elif len(cfg.simulation_legs) == 1:
                        leg1 = cfg.simulation_legs[0]
                        dg = leg_results[leg1][0]
                        dg_err = leg_results[leg1][1]
                        print('dG = {}: SEM = {}'.format(dg, dg_err))
                        result[engine_id][prot_name][lig_name] = [dg, dg_err]

                    else:
                        for leg in cfg.simulation_legs:
                            print('result = {} SEM = {} for leg {}'.format(*leg_results[leg], leg))
                        fin_result = {leg: leg_results[leg] for leg in cfg.simulation_legs}
                        result[engine_id][prot_name][lig_name] = fin_result

        with open('./result.dat', 'w') as f:
            print(result, file=f)


def nice_print(string):
    '''

    :param string: str to format
    :return: str, formatted string
    '''
    string = string.center(75, '#')
    print(string)


def main():
    nice_print('Analysis')

    if os.path.exists('./result.dat'):
        raise ValueError('Results file found in this directory cowardly refusing to proceed.')
    else:
        ana = Analysis()
