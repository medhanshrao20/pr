"""Brentwood-only graph extension of the copied L.1 architecture."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.experiment import main
if __name__=="__main__": main("spatial_L.1")
