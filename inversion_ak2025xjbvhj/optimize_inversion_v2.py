#!/usr/bin/env python
"""
Optimized moment tensor inversion - Version 2
Using hybrid approach with station quality control
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

# Read filtered weights
weights_filtered = os.path.join(script_dir, 'weights_filtered.dat')

print('='*70)
print('OPTIMIZED MOMENT TENSOR INVERSION v2')
print('='*70)

# Process parameters optimized for deep Alaska earthquake
# Use longer periods to better match 1D model at regional distances
process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.02,       # 50 second period
    freq_max=0.05,       # 20 second period - very long period for stability
    pick_type='user_supplied',
    window_type='surface_wave',
    window_length=150.,
    capuaf_file=weights_filtered,
)

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.02,       # 50 second period  
    freq_max=0.08,       # 12.5 second period
    pick_type='user_supplied',
    window_type='body_wave',
    window_length=30.,
    capuaf_file=weights_filtered,
)

# L2 misfit with generous time shifts to allow for 1D model errors
misfit_sw = Misfit(
    norm='L2',
    time_shift_min=-15.,
    time_shift_max=+15.,
    time_shift_groups=['ZR', 'T'],
)

misfit_bw = Misfit(
    norm='L2',
    time_shift_min=-5.,
    time_shift_max=+5.,
    time_shift_groups=['ZR'],
)

# Read data
print('\nReading data...')
station_id_list = parse_station_codes(weights_filtered)
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

# Test multiple magnitudes
magnitude_tests = [5.9, 6.0, 6.1, 6.2]
best_overall_misfit = float('inf')
best_overall_mt = None
best_overall_lune = None
best_overall_results = None
best_magnitude = None

for test_mag in magnitude_tests:
    wavelet = Trapezoid(magnitude=test_mag)
    greens_copy = greens.copy()
    greens_copy.convolve(wavelet)
    greens_bw = greens_copy.map(process_bw)
    greens_sw = greens_copy.map(process_sw)
    
    # Single stage grid search
    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=12,
        magnitudes=[test_mag],
    )
    
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
    
    # Combine with surface wave emphasis (deep event)
    results = 0.33 * results_bw + 0.67 * results_sw
    
    min_misfit = float(results.min())
    print(f'  Mw {test_mag}: misfit = {min_misfit:.6f}')
    
    if min_misfit < best_overall_misfit:
        best_overall_misfit = min_misfit
        best_magnitude = test_mag
        idx = results.source_idxmin()
        best_overall_mt = grid.get(idx)
        best_overall_lune = grid.get_dict(idx)
        best_overall_results = results

print(f'\nBest magnitude: Mw {best_magnitude}')

# Refine around best solution
print('\nRefining solution...')
wavelet = Trapezoid(magnitude=best_magnitude)
greens_copy = greens.copy()
greens_copy.convolve(wavelet)
greens_bw = greens_copy.map(process_bw)
greens_sw = greens_copy.map(process_sw)

# Fine grid
grid_fine = FullMomentTensorGridSemiregular(
    npts_per_axis=20,
    magnitudes=[best_magnitude - 0.05, best_magnitude, best_magnitude + 0.05],
)

results_sw_fine = grid_search(data_sw, greens_sw, misfit_sw, origin, grid_fine)
results_bw_fine = grid_search(data_bw, greens_bw, misfit_bw, origin, grid_fine)
results_fine = 0.33 * results_bw_fine + 0.67 * results_sw_fine

idx = results_fine.source_idxmin()
best_mt = grid_fine.get(idx)
lune_dict = grid_fine.get_dict(idx)
mt_dict = best_mt.as_dict()

# Calculate variance reduction
def calc_variance_reduction(data_list, greens_list, mt, misfit_obj):
    """Calculate total variance reduction across all stations"""
    total_data_energy = 0
    total_residual_energy = 0
    
    for i in range(len(data_list)):
        if data_list[i] is None or len(data_list[i]) == 0:
            continue
        try:
            d = data_list[i]
            g = greens_list[i]
            components = [tr.stats.channel[-1] for tr in d]
            g._set_components(components)
            s = g.get_synthetics(mt, components)
            
            for j, (tr_d, tr_s) in enumerate(zip(d, s)):
                dd = tr_d.data
                ss = tr_s.data
                
                # Apply time shift
                from mtuq.misfit.waveform.level0 import _xcorr_shift
                min_len = min(len(dd), len(ss))
                shift, _ = _xcorr_shift(dd[:min_len], ss[:min_len], 
                                        int(misfit_obj.time_shift_max / tr_d.stats.delta))
                if abs(shift) < len(ss):
                    if shift >= 0:
                        ss_shifted = ss[shift:shift+min_len] if shift+min_len <= len(ss) else ss[shift:]
                        dd_crop = dd[:len(ss_shifted)]
                    else:
                        dd_crop = dd[-shift:-shift+min_len] if -shift+min_len <= len(dd) else dd[-shift:]
                        ss_shifted = ss[:len(dd_crop)]
                    
                    min_len = min(len(dd_crop), len(ss_shifted))
                    if min_len > 0:
                        total_data_energy += np.sum(dd_crop[:min_len]**2)
                        total_residual_energy += np.sum((dd_crop[:min_len] - ss_shifted[:min_len])**2)
        except Exception as e:
            continue
    
    if total_data_energy > 0:
        return 1.0 - total_residual_energy / total_data_energy
    return 0.0

print('\nCalculating variance reduction...')
vr_sw = calc_variance_reduction(data_sw, greens_sw, best_mt, misfit_sw)
vr_bw = calc_variance_reduction(data_bw, greens_bw, best_mt, misfit_bw)
combined_vr = 0.33 * vr_bw + 0.67 * vr_sw

# Print results
print('\n' + '='*70)
print('OPTIMIZED RESULTS v2')
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
print(f'Final misfit: {float(results_fine.min()):.6f}')
print()
print(f'Variance Reduction:')
print(f'  Surface waves: {vr_sw:.4f} ({vr_sw*100:.1f}%)')
print(f'  Body waves: {vr_bw:.4f} ({vr_bw*100:.1f}%)')
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
output_dir = os.path.join(script_dir, 'results_optimized_v2')
os.makedirs(output_dir, exist_ok=True)

print('\nGenerating output files...')

try:
    plot_data_greens2(
        os.path.join(output_dir, f'{event_id}_FMT_optimized_v2_waveforms.png'),
        data_bw, data_sw, greens_bw, greens_sw, 
        process_bw, process_sw, misfit_bw, misfit_sw, 
        stations, origin, best_mt, lune_dict
    )
    print('  Created waveform plot')
except Exception as e:
    print(f'  Warning: waveform plot failed: {e}')

try:
    plot_beachball(
        os.path.join(output_dir, f'{event_id}_FMT_optimized_v2_beachball.png'),
        best_mt, stations, origin
    )
    print('  Created beachball plot')
except Exception as e:
    print(f'  Warning: beachball plot failed: {e}')

try:
    plot_misfit_lune(
        os.path.join(output_dir, f'{event_id}_FMT_optimized_v2_misfit_lune.png'),
        results_fine
    )
    print('  Created misfit lune plot')
except Exception as e:
    print(f'  Warning: misfit lune plot failed: {e}')

save_json(
    os.path.join(output_dir, f'{event_id}_FMT_optimized_v2_solution.json'),
    merged_dict
)
print(f'  Saved solution JSON')

# Save summary
with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
    f.write('OPTIMIZED MOMENT TENSOR INVERSION v2\n')
    f.write('='*50 + '\n\n')
    f.write(f'Event: {event_id}\n')
    f.write(f'Stations used: {len(stations)}\n')
    f.write(f'Processing: Long period (20-50s SW, 12.5-50s BW)\n\n')
    f.write(f'Results:\n')
    f.write(f'  Mw: {best_mt.magnitude():.2f}\n')
    f.write(f'  Strike: {lune_dict["kappa"]:.1f}°\n')
    f.write(f'  Slip: {lune_dict["sigma"]:.1f}°\n')
    f.write(f'  Dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°\n\n')
    f.write(f'Lune coordinates:\n')
    f.write(f'  v: {lune_dict["v"]:.3f}\n')
    f.write(f'  w: {lune_dict["w"]:.3f}\n\n')
    f.write(f'Misfit: {float(results_fine.min()):.6f}\n\n')
    f.write(f'Variance Reduction:\n')
    f.write(f'  SW: {vr_sw*100:.1f}%\n')
    f.write(f'  BW: {vr_bw*100:.1f}%\n')
    f.write(f'  Combined: {combined_vr*100:.1f}%\n')

print(f'\nResults saved to: {output_dir}')
print('\n' + '='*70)
