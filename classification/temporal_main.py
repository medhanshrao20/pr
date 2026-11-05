"""Run the new local temporal classification arm independently."""
import sys
from run_experiment import ROOT, main

if __name__ == '__main__':
    main(['--conditions', 'temporal', '--output', str(ROOT / 'results/temporal_classification'), *sys.argv[1:]])
