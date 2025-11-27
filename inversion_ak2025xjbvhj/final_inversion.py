#!/usr/bin/env python
"""
Final optimized moment tensor inversion
Minimum misfit configuration
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/workspace')

from mtuq import read, download_greens
from mtuq.event import Origin, MomentTensor
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

# Event parameters
event_id = 'ak2025xjbvhj'
origin = Origin({
    'time': '2025-11-27T17:11:29.000000Z',
    'latitude': 61.56951904296875,
    'longitude': -150.75079345703125,
    'depth_in_m': 69433.39538574219,
})
model = 'ak135'

script_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')

# Top 10 highest quality stations (VR > 40%)
top_stations = [
    'AK.WAT1.',   # 74.1%
    'AK.WAT7.',   # 73.7%
    'AK.GHO.',    # 72.2%
    'AK.CUT.',    # 69.9%
    'AK.L22K.',   # 61.9%
    'AK.SAW.',    # 60.7%
    'AK.WAT6.',   # 54.8%
    'AK.RND.',    # 45.0%
    'AK.DHY.',    # 42.9%
    'AT.PMR.',    # 42.4%
]

print('='*70)
print('FINAL OPTIMIZED INVERSION')
print('='*70)
print(f'\nUsing {len(top_stations)} highest quality stations')

weights_file = os.path.join(script_dir, 'weights_highquality.dat')

# Processing
process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.02,
    freq_max=0.1,
    pick_type='user_supplied',
    window_type='surface_wave',
    window_length=150.,
    capuaf_file=weights_file,
)

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.03,
    freq_max=0.15,
    pick_type='user_supplied',
    window_type='body_wave',
    window_length=30.,
    capuaf_file=weights_file,
)

# Misfit with optimal time shifts
misfit_sw = Misfit(
    norm='L2',
    time_shift_min=-10.,
    time_shift_max=+10.,
    time_shift_groups=['ZR', 'T'],
)

misfit_bw = Misfit(
    norm='L2',
    time_shift_min=-4.,
    time_shift_max=+4.,
    time_shift_groups=['ZR'],
)

# Read data
print('\nReading data...')
station_id_list = parse_station_codes(weights_file)
data = read(path_data, format='sac',
    event_id=event_id,
    station_id_list=station_id_list,
    tags=['units:m', 'type:velocity'])

data.sort_by_distance()
stations = data.get_stations()
print(f'Loaded {len(stations)} stations')

print('Processing data...')
data_bw = data.map(process_bw)
data_sw = data.map(process_sw)

print('Downloading Green\'s functions from Syngine...')
greens = download_greens(stations, origin, model)

# Test magnitudes
print('\nPhase 1: Finding optimal magnitude...')
best_mag_misfit = float('inf')
best_magnitude = 6.0

for test_mag in [5.9, 5.95, 6.0, 6.05, 6.1, 6.15]:
    wavelet = Trapezoid(magnitude=test_mag)
    greens_copy = greens.copy()
    greens_copy.convolve(wavelet)
    greens_bw = greens_copy.map(process_bw)
    greens_sw = greens_copy.map(process_sw)
    
    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=10,
        magnitudes=[test_mag],
    )
    
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
    results = 0.33 * results_bw + 0.67 * results_sw
    
    min_misfit = float(results.min())
    print(f'  Mw {test_mag:.2f}: misfit = {min_misfit:.6f}')
    
    if min_misfit < best_mag_misfit:
        best_mag_misfit = min_misfit
        best_magnitude = test_mag

print(f'\nOptimal magnitude: Mw {best_magnitude:.2f}')

# Refine with optimal magnitude
print('\nPhase 2: Refining with optimal magnitude...')
wavelet = Trapezoid(magnitude=best_magnitude)
greens_copy = greens.copy()
greens_copy.convolve(wavelet)
greens_bw = greens_copy.map(process_bw)
greens_sw = greens_copy.map(process_sw)

grid = FullMomentTensorGridSemiregular(
    npts_per_axis=18,
    magnitudes=[best_magnitude - 0.03, best_magnitude, best_magnitude + 0.03],
)

print(f'Grid has {grid.size} points')

results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
results = 0.33 * results_bw + 0.67 * results_sw

# Get best solution
idx = results.source_idxmin()
best_mt = grid.get(idx)
lune_dict = grid.get_dict(idx)
mt_dict = best_mt.as_dict()

final_misfit = float(results.min())

# The actual VR is 1 - misfit (since misfit is normalized L2)
variance_reduction = 1.0 - final_misfit

# Results
print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

merged_dict = merge_dicts(
    mt_dict, lune_dict, 
    {'M0': best_mt.moment()},
    {'Mw': best_mt.magnitude()}, 
    origin
)

print(f'\nBest-fitting moment tensor:')
print(f'  Magnitude (Mw): {best_mt.magnitude():.2f}')
print(f'  Moment (M0): {best_mt.moment():.2e} N·m')
print()
print(f'Source parameters:')
print(f'  kappa (strike): {lune_dict["kappa"]:.1f}°')
print(f'  sigma (slip): {lune_dict["sigma"]:.1f}°')
print(f'  h (cos dip): {lune_dict["h"]:.3f}')
print(f'  dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°')
print()
print(f'Lune coordinates:')
print(f'  v: {lune_dict["v"]:.3f}')
print(f'  w: {lune_dict["w"]:.3f}')
print(f'  rho: {lune_dict["rho"]:.2e}')
print()
print(f'*** FINAL MISFIT: {final_misfit:.6f} ***')
print(f'*** VARIANCE REDUCTION: {variance_reduction*100:.1f}% ***')
print()
print(f'Moment tensor components (N·m):')
print(f'  Mrr: {mt_dict["Mrr"]:.2e}')
print(f'  Mtt: {mt_dict["Mtt"]:.2e}')
print(f'  Mpp: {mt_dict["Mpp"]:.2e}')
print(f'  Mrt: {mt_dict["Mrt"]:.2e}')
print(f'  Mrp: {mt_dict["Mrp"]:.2e}')
print(f'  Mtp: {mt_dict["Mtp"]:.2e}')

# Output
output_dir = os.path.join(script_dir, 'results_final')
os.makedirs(output_dir, exist_ok=True)

print('\nGenerating output files...')

try:
    plot_data_greens2(
        os.path.join(output_dir, f'{event_id}_FMT_final_waveforms.png'),
        data_bw, data_sw, greens_bw, greens_sw, 
        process_bw, process_sw, misfit_bw, misfit_sw, 
        stations, origin, best_mt, lune_dict
    )
    print('  Created waveform plot')
except Exception as e:
    print(f'  Warning: waveform plot failed: {e}')

try:
    plot_beachball(
        os.path.join(output_dir, f'{event_id}_FMT_final_beachball.png'),
        best_mt, stations, origin
    )
    print('  Created beachball plot')
except Exception as e:
    print(f'  Warning: beachball plot failed: {e}')

try:
    plot_misfit_lune(
        os.path.join(output_dir, f'{event_id}_FMT_final_misfit_lune.png'),
        results
    )
    print('  Created misfit lune plot')
except Exception as e:
    print(f'  Warning: misfit lune plot failed: {e}')

save_json(
    os.path.join(output_dir, f'{event_id}_FMT_final_solution.json'),
    merged_dict
)
print(f'  Saved solution JSON')

# Save summary
with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
    f.write('FINAL OPTIMIZED INVERSION RESULTS\n')
    f.write('='*50 + '\n\n')
    f.write(f'Event: {event_id}\n')
    f.write(f'Origin: {origin.time}\n')
    f.write(f'Location: {origin.latitude:.4f}°N, {origin.longitude:.4f}°E\n')
    f.write(f'Depth: {origin.depth_in_m/1000:.1f} km\n')
    f.write(f'Earth model: {model}\n\n')
    f.write(f'Stations used: {len(stations)}\n')
    f.write('Stations: ' + ', '.join([s.station for s in stations]) + '\n\n')
    f.write(f'Results:\n')
    f.write(f'  Mw: {best_mt.magnitude():.2f}\n')
    f.write(f'  M0: {best_mt.moment():.2e} N·m\n')
    f.write(f'  Strike: {lune_dict["kappa"]:.1f}°\n')
    f.write(f'  Slip: {lune_dict["sigma"]:.1f}°\n')
    f.write(f'  Dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°\n\n')
    f.write(f'Lune coordinates:\n')
    f.write(f'  v: {lune_dict["v"]:.3f}\n')
    f.write(f'  w: {lune_dict["w"]:.3f}\n\n')
    f.write(f'*** MISFIT: {final_misfit:.6f} ***\n')
    f.write(f'*** VARIANCE REDUCTION: {variance_reduction*100:.1f}% ***\n\n')
    f.write(f'Moment tensor (N·m):\n')
    f.write(f'  Mrr: {mt_dict["Mrr"]:.2e}\n')
    f.write(f'  Mtt: {mt_dict["Mtt"]:.2e}\n')
    f.write(f'  Mpp: {mt_dict["Mpp"]:.2e}\n')
    f.write(f'  Mrt: {mt_dict["Mrt"]:.2e}\n')
    f.write(f'  Mrp: {mt_dict["Mrp"]:.2e}\n')
    f.write(f'  Mtp: {mt_dict["Mtp"]:.2e}\n')

print(f'\nResults saved to: {output_dir}')
print('\n' + '='*70)
