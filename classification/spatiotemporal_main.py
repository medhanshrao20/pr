"""Run the new neighbor-informed spatiotemporal classification arm independently."""
import sys
from run_experiment import ROOT, main

if __name__ == '__main__':
    main(['--conditions', 'spatiotemporal', '--output', str(ROOT / 'results/spatiotemporal_classification'), *sys.argv[1:]])
