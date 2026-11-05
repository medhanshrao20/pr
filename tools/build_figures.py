"""Scientific figures from saved classification evidence; no model fitting."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results/classification'
info = json.loads((R / 'metadata.json').read_text())
with (R / 'paired_comparison.csv').open() as f:
    paired = list(csv.DictReader(f))
delta = np.array([float(r['test_macro_f1_difference']) for r in paired])
lower = np.array([float(r['macro_f1_difference_ci_low']) for r in paired])
upper = np.array([float(r['macro_f1_difference_ci_high']) for r in paired])
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.errorbar(delta, range(len(paired)), xerr=np.vstack([delta - lower, upper - delta]), fmt='o', capsize=5, color='#176b82')
ax.axvline(0, color='#777777', linewidth=1)
ax.set_yticks(range(len(paired)), [r['model'] for r in paired])
ax.set_xlabel('Spatiotemporal minus temporal macro-F1')
ax.set_title('2025 paired day-bootstrap 95% intervals')
fig.tight_layout(); fig.savefig(R / 'paired_macro_f1.png', dpi=180); plt.close(fig)
coordinates = {r['id']: r for r in json.loads((ROOT / 'data/station_coordinates.json').read_text())}
fig, ax = plt.subplots(figsize=(7, 3.6))
for sid in [47] + [n['station'] for n in info['neighbors']]:
    q = coordinates[sid]
    ax.scatter(q['lon'], q['lat'], s=80, marker='*' if sid == 47 else 'o', color='#a55035' if sid == 47 else '#176b82')
    ax.annotate(q['name'], (q['lon'], q['lat']), xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.margins(.23); ax.set_xlabel('Longitude (degrees)'); ax.set_ylabel('Latitude (degrees)')
ax.set_title('Target and fixed geographically nearest neighbors')
fig.tight_layout(); fig.savefig(R / 'neighbor_layout.png', dpi=170); plt.close(fig)
print('Scientific diagnostic figures generated from saved evidence.')
