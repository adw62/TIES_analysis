#!/usr/bin/env python

import unittest
import numpy as np
from ties_analysis.engines.openmm import *

class Test_OpenMM(unittest.TestCase):

    def test_collating(self):

        vdw_a=[]
        vdw_d=[]
        ele_a=[]
        ele_d=[]

        openmm_fep = OpenMM('FEP', None, None, vdw_a, vdw_d, ele_a, ele_d, [False])
        openmm_ti = OpenMM('TI', None, None, vdw_a, vdw_d, ele_a, ele_d, [False])

        data_dir = './data/OpenMM/test_collate/ethane_zero/leg1'

        fep_ans = np.load(os.path.join(data_dir, 'assembled_fep.npy'))
        ti_ans = np.load(os.path.join(data_dir, 'assembled_ti.npy'))

        fep_built = openmm_fep.collate_data('./data', 'test_collate', 'ethane_zero', 'leg1')
        ti_built = openmm_ti.collate_data('./data', 'test_collate', 'ethane_zero', 'leg1')

        self.assertIsNone(np.testing.assert_array_equal(fep_ans, fep_built, 'FEP result built incorrectly for OpenMM'))
        self.assertIsNone(np.testing.assert_array_equal(ti_ans, ti_built, 'TI result built incorrectly for OpenMM'))

if __name__ == '__main__':
    unittest.main()