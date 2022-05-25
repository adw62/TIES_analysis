#!/usr/bin/env python

from ties_analysis.ties_analysis import Analysis
from ties_analysis.config import Config

cfg = Config()
cfg.rep_convg = []
print('Running with options:')
cfg.get_options()
ana = Analysis(cfg)
ana.run()

