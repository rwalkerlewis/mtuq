#!/usr/bin/env python
"""
Optimized moment tensor inversion with:
1. Stricter station selection (VR > 0.1)
2. Adjusted frequency bands
3. Finer grid search
4. Optimized weighting
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
magnitude = 6.0

script_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')
path_weights_orig = os.path.join(script_dir, 'weights_filtered.dat')

# Best stations from previous analysis (VR > 0.15)
best_stations = [
    'AV.STLK.', 'AK.WAT1.', 'AK.WAT7.', 'AK.GHO.', 'AK.CUT.', 
    'AK.L22K.', 'AK.WAT6.', 'AK.SAW.', 'AK.RND.', 'AK.DHY.',
    'AT.PMR.', 'AK.SCM.', 'AK.M23K.', 'AK.PWL.', 'AK.BAE.',
    'AK.CAST.', 'AV.RDSO.', 'AV.RDJH.', 'AV.RDDF.', 'AV.SPCG.',
    'AV.RDT.', 'AV.RED.', 'AV.RDWB.', 'AK.KNK.', 'AV.REF.',
    'AK.BRLK.', 'AV.SPCN.', 'AK.BRSE.', 'AV.SPU.', 'AV.NCT.',
]

print('='*70)
print('OPTIMIZED MOMENT TENSOR INVERSION')
print('='*70)
print(f'\nUsing {len(best_stations)} best-fitting stations (VR > 0.1)')

# Create optimized weights file
weights_optimized = os.path.join(script_dir, 'weights_optimized.dat')
with open(path_weights_orig, 'r') as f:
    lines = f.readlines()

with open(weights_optimized, 'w') as f:
    for line in lines:
        if line.startswith('#'):
            f.write(line)
            continue
        parts = line.split()
        if len(parts) > 0:
            station_id = '.'.join(parts[0].split('.')[1:4])
            if station_id in best_stations:
                f.write(line)

print(f'Created optimized weights file: {weights_optimized}')

# Optimized processing - narrower frequency band for cleaner signal
process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.025,      # 40 second period
    freq_max=0.08,       # 12.5 second period - narrower band
    pick_type='user_supplied',
    window_type='surface_wave',
    window_length=120.,  # Shorter window
    capuaf_file=weights_optimized,
)

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.04,       # 25 second period  
    freq_max=0.12,       # 8.3 second period
    pick_type='user_supplied',
    window_type='body_wave',
    window_length=25.,
    capuaf_file=weights_optimized,
)

# Misfit with optimized time shifts
misfit_sw = Misfit(
    norm='L2',
    time_shift_min=-12.,
    time_shift_max=+12.,
    time_shift_groups=['ZR', 'T'],
    normalize=True,
)

misfit_bw = Misfit(
    norm='L2',
    time_shift_min=-4.,
    time_shift_max=+4.,
    time_shift_groups=['ZR'],
    normalize=True,
)

# Read data
print('\nReading data...')
station_id_list = parse_station_codes(weights_optimized)
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

wavelet = Trapezoid(magnitude=magnitude)
greens.convolve(wavelet)
greens_bw = greens.map(process_bw)
greens_sw = greens.map(process_sw)

# Finer grid search around the known solution
print('\nPhase 1: Coarse grid search...')
grid_coarse = FullMomentTensorGridSemiregular(
    npts_per_axis=10,
    magnitudes=[5.9, 6.0, 6.1, 6.2],
)

results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid_coarse)
results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid_coarse)

# Try different weightings
best_misfit = float('inf')
best_weight = 0.5
for sw_weight in [0.5, 0.6, 0.7, 0.8]:
    results = (1-sw_weight) * results_bw + sw_weight * results_sw
    min_misfit = float(results.min())
    if min_misfit < best_misfit:
        best_misfit = min_misfit
        best_weight = sw_weight

print(f'Optimal SW weight: {best_weight}')
results = (1-best_weight) * results_bw + best_weight * results_sw

# Get best solution from coarse search
idx = results.source_idxmin()
best_mt_coarse = grid_coarse.get(idx)
lune_dict_coarse = grid_coarse.get_dict(idx)

print(f'\nCoarse solution: Mw={best_mt_coarse.magnitude():.2f}')
print(f'  kappa={lune_dict_coarse["kappa"]:.0f}, sigma={lune_dict_coarse["sigma"]:.0f}, h={lune_dict_coarse["h"]:.2f}')

# Phase 2: Fine grid search around best solution
print('\nPhase 2: Fine grid search around best solution...')
grid_fine = FullMomentTensorGridSemiregular(
    npts_per_axis=15,  # Finer grid
    magnitudes=[best_mt_coarse.magnitude()-0.1, best_mt_coarse.magnitude(), best_mt_coarse.magnitude()+0.1],
)

results_sw_fine = grid_search(data_sw, greens_sw, misfit_sw, origin, grid_fine)
results_bw_fine = grid_search(data_bw, greens_bw, misfit_bw, origin, grid_fine)
results_fine = (1-best_weight) * results_bw_fine + best_weight * results_sw_fine

# Final result
idx = results_fine.source_idxmin()
best_mt = grid_fine.get(idx)
lune_dict = grid_fine.get_dict(idx)
mt_dict = best_mt.as_dict()

# Calculate variance reduction for the final solution
print('\nCalculating final variance reduction...')
total_vr_sw = []
total_vr_bw = []

for i, station in enumerate(stations):
    try:
        # Surface wave VR
        if i < len(data_sw) and data_sw[i] is not None and len(data_sw[i]) > 0:
            d_sw = data_sw[i]
            g_sw = greens_sw[i]
            components = [tr.stats.channel[-1] for tr in d_sw]
            g_sw._set_components(components)
            s_sw = g_sw.get_synthetics(best_mt)
            
            d_energy = 0
            res_energy = 0
            for j, tr in enumerate(d_sw):
                d = tr.data
                if j < len(s_sw):
                    s = s_sw[j].data
                    min_len = min(len(d), len(s))
                    d_energy += np.sum(d[:min_len]**2)
                    res_energy += np.sum((d[:min_len] - s[:min_len])**2)
            if d_energy > 0:
                total_vr_sw.append(1.0 - res_energy/d_energy)
        
        # Body wave VR
        if i < len(data_bw) and data_bw[i] is not None and len(data_bw[i]) > 0:
            d_bw = data_bw[i]
            g_bw = greens_bw[i]
            components = [tr.stats.channel[-1] for tr in d_bw]
            g_bw._set_components(components)
            s_bw = g_bw.get_synthetics(best_mt)
            
            d_energy = 0
            res_energy = 0
            for j, tr in enumerate(d_bw):
                d = tr.data
                if j < len(s_bw):
                    s = s_bw[j].data
                    min_len = min(len(d), len(s))
                    d_energy += np.sum(d[:min_len]**2)
                    res_energy += np.sum((d[:min_len] - s[:min_len])**2)
            if d_energy > 0:
                total_vr_bw.append(1.0 - res_energy/d_energy)
    except:
        pass

mean_vr_sw = np.mean(total_vr_sw) if total_vr_sw else 0
mean_vr_bw = np.mean(total_vr_bw) if total_vr_bw else 0
combined_vr = best_weight * mean_vr_sw + (1-best_weight) * mean_vr_bw

# Results
print('\n' + '='*70)
print('OPTIMIZED RESULTS')
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
print(f'Fault parameters:')
print(f'  kappa (strike): {lune_dict["kappa"]:.1f}°')
print(f'  sigma (slip): {lune_dict["sigma"]:.1f}°')
print(f'  h (cos dip): {lune_dict["h"]:.3f}')
print(f'  dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°')
print()
print(f'Variance Reduction:')
print(f'  Surface waves: {mean_vr_sw:.4f} ({mean_vr_sw*100:.1f}%)')
print(f'  Body waves: {mean_vr_bw:.4f} ({mean_vr_bw*100:.1f}%)')
print(f'  Combined (weighted): {combined_vr:.4f} ({combined_vr*100:.1f}%)')
print()
print(f'Moment tensor components (N·m):')
print(f'  Mrr: {mt_dict["Mrr"]:.2e}')
print(f'  Mtt: {mt_dict["Mtt"]:.2e}')
print(f'  Mpp: {mt_dict["Mpp"]:.2e}')
print(f'  Mrt: {mt_dict["Mrt"]:.2e}')
print(f'  Mrp: {mt_dict["Mrp"]:.2e}')
print(f'  Mtp: {mt_dict["Mtp"]:.2e}')

# Output
output_dir = os.path.join(script_dir, 'results_optimized')
os.makedirs(output_dir, exist_ok=True)

print('\nGenerating output files...')

try:
    plot_data_greens2(
        os.path.join(output_dir, f'{event_id}_FMT_optimized_waveforms.png'),
        data_bw, data_sw, greens_bw, greens_sw, 
        process_bw, process_sw, misfit_bw, misfit_sw, 
        stations, origin, best_mt, lune_dict
    )
    print('  Created waveform plot')
except Exception as e:
    print(f'  Warning: waveform plot failed: {e}')

try:
    plot_beachball(
        os.path.join(output_dir, f'{event_id}_FMT_optimized_beachball.png'),
        best_mt, stations, origin
    )
    print('  Created beachball plot')
except Exception as e:
    print(f'  Warning: beachball plot failed: {e}')

try:
    plot_misfit_lune(
        os.path.join(output_dir, f'{event_id}_FMT_optimized_misfit_lune.png'),
        results_fine
    )
    print('  Created misfit lune plot')
except Exception as e:
    print(f'  Warning: misfit lune plot failed: {e}')

save_json(
    os.path.join(output_dir, f'{event_id}_FMT_optimized_solution.json'),
    merged_dict
)
print(f'  Saved solution JSON')

# Save summary
with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
    f.write('OPTIMIZED MOMENT TENSOR INVERSION\n')
    f.write('='*50 + '\n\n')
    f.write(f'Event: {event_id}\n')
    f.write(f'Stations used: {len(stations)}\n')
    f.write(f'SW weight: {best_weight}\n\n')
    f.write(f'Results:\n')
    f.write(f'  Mw: {best_mt.magnitude():.2f}\n')
    f.write(f'  Strike: {lune_dict["kappa"]:.1f}°\n')
    f.write(f'  Slip: {lune_dict["sigma"]:.1f}°\n')
    f.write(f'  Dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°\n\n')
    f.write(f'Variance Reduction:\n')
    f.write(f'  SW: {mean_vr_sw*100:.1f}%\n')
    f.write(f'  BW: {mean_vr_bw*100:.1f}%\n')
    f.write(f'  Combined: {combined_vr*100:.1f}%\n')

print(f'\nResults saved to: {output_dir}')
print('\n' + '='*70)
print('OPTIMIZATION COMPLETE')
print('='*70)
