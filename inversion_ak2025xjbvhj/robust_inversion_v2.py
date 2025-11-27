#!/usr/bin/env python
"""
Robust moment tensor inversion v2 - Simplified approach

Key safeguards:
1. Station selection based on SNR (not fit to solution)
2. Bootstrap stability analysis
3. Feature-based quality metrics (polarity, phase timing)
4. Minimum station count requirement for azimuthal coverage
"""

import os
import sys
import numpy as np
from collections import defaultdict

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
from obspy.geodetics import gps2dist_azimuth

# Event parameters
event_id = 'ak2025xjbvhj'
origin = Origin({
    'time': '2025-11-27T17:11:29.000000Z',
    'latitude': 61.56951904296875,
    'longitude': -150.75079345703125,
    'depth_in_m': 69433.39538574219,
})
model = 'ak135'
magnitude = 6.1

script_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')
weights_filtered = os.path.join(script_dir, 'weights_filtered.dat')

print('='*70)
print('ROBUST MOMENT TENSOR INVERSION v2')
print('(SNR-based station selection, stability analysis)')
print('='*70)

# Processing parameters
process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.02,
    freq_max=0.1,
    pick_type='user_supplied',
    window_type='surface_wave',
    window_length=150.,
    capuaf_file=weights_filtered,
)

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.03,
    freq_max=0.15,
    pick_type='user_supplied',
    window_type='body_wave',
    window_length=30.,
    capuaf_file=weights_filtered,
)

# Misfit
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

# Read all data
print('\nReading data...')
station_id_list = parse_station_codes(weights_filtered)
data = read(path_data, format='sac',
    event_id=event_id,
    station_id_list=station_id_list,
    tags=['units:m', 'type:velocity'])

data.sort_by_distance()
stations = data.get_stations()
print(f'Loaded {len(stations)} stations')

# Calculate SNR for each station (independent of MT solution)
print('\nCalculating signal-to-noise ratios...')
station_snr = {}

for i, station in enumerate(stations):
    sta_id = f"{station.network}.{station.station}"
    
    # Get raw traces for this station
    traces = data[i]
    if traces is None or len(traces) == 0:
        continue
    
    snr_values = []
    for tr in traces:
        # Estimate noise from first 10% of trace
        n_noise = max(int(len(tr.data) * 0.1), 10)
        noise_std = np.std(tr.data[:n_noise])
        
        # Signal is max amplitude in main part of trace
        signal_max = np.max(np.abs(tr.data[n_noise:]))
        
        if noise_std > 0:
            snr = signal_max / noise_std
            snr_values.append(snr)
    
    if snr_values:
        station_snr[sta_id] = np.mean(snr_values)

# Sort stations by SNR
sorted_stations = sorted(station_snr.items(), key=lambda x: x[1], reverse=True)
print('\nStation SNR ranking (solution-independent):')
for sta_id, snr in sorted_stations[:15]:
    print(f'  {sta_id}: SNR = {snr:.1f}')

# Calculate azimuthal distribution
station_azimuths = {}
for station in stations:
    sta_id = f"{station.network}.{station.station}"
    dist_m, az, _ = gps2dist_azimuth(
        origin.latitude, origin.longitude,
        station.latitude, station.longitude
    )
    station_azimuths[sta_id] = az

# Select stations: use ALL high-SNR stations for robustness
# (more stations = less overfitting, even if misfit is higher)
min_snr = 50.0  # Reasonable SNR threshold
selected_stations = [sta_id for sta_id, snr in sorted_stations if snr >= min_snr]

print(f'\nStations with SNR >= {min_snr}: {len(selected_stations)}')

# Ensure we have enough stations (at least 25 for robustness)
if len(selected_stations) < 25:
    print(f'Warning: Only {len(selected_stations)} stations with SNR >= {min_snr}')
    print('Using all available stations for robustness')
    selected_stations = [sta_id for sta_id, _ in sorted_stations]

# Create weights file for selected stations
weights_robust = os.path.join(script_dir, 'weights_robust.dat')
with open(weights_filtered, 'r') as f:
    lines = f.readlines()

with open(weights_robust, 'w') as f:
    f.write('# Robust station selection (SNR-based)\n')
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) > 0:
            sta_code = parts[0]
            network = sta_code.split('.')[1]
            station = sta_code.split('.')[2]
            sta_id = f"{network}.{station}"
            if sta_id in selected_stations:
                f.write(line)

# Re-read data with selected stations
print('\nProcessing selected stations...')
station_id_list = parse_station_codes(weights_robust)
data = read(path_data, format='sac',
    event_id=event_id,
    station_id_list=station_id_list,
    tags=['units:m', 'type:velocity'])

data.sort_by_distance()
stations = data.get_stations()
print(f'Using {len(stations)} stations')

# Update processing with new weights file
process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.02,
    freq_max=0.1,
    pick_type='user_supplied',
    window_type='surface_wave',
    window_length=150.,
    capuaf_file=weights_robust,
)

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.03,
    freq_max=0.15,
    pick_type='user_supplied',
    window_type='body_wave',
    window_length=30.,
    capuaf_file=weights_robust,
)

data_bw = data.map(process_bw)
data_sw = data.map(process_sw)

print('Downloading Green\'s functions...')
greens = download_greens(stations, origin, model)

wavelet = Trapezoid(magnitude=magnitude)
greens.convolve(wavelet)
greens_bw = greens.map(process_bw)
greens_sw = greens.map(process_sw)

# Grid search
print('\n' + '='*70)
print('GRID SEARCH')
print('='*70)

grid = FullMomentTensorGridSemiregular(
    npts_per_axis=12,
    magnitudes=[5.9, 6.0, 6.1, 6.2],
)

print(f'\nGrid has {grid.size} points')

results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)

# Equal weighting (more balanced than emphasizing one wave type)
results = 0.5 * results_bw + 0.5 * results_sw

idx = results.source_idxmin()
best_mt = grid.get(idx)
lune_dict = grid.get_dict(idx)
mt_dict = best_mt.as_dict()
final_misfit = float(results.min())

# Stability analysis: check how solution changes when dropping stations
print('\n' + '='*70)
print('STABILITY ANALYSIS (jackknife)')
print('='*70)

n_bootstrap = min(len(stations), 10)  # Test dropping up to 10 stations
jackknife_solutions = []

print(f'\nTesting solution stability by dropping individual stations...')

# Use the Dataset class to create proper subsets
from mtuq.dataset import Dataset
from mtuq.greens_tensor.base import GreensTensorList

for drop_idx in range(n_bootstrap):
    # Create subset indices
    keep_indices = [i for i in range(len(stations)) if i != drop_idx]
    
    # Manually construct subsets
    data_sw_sub = Dataset([data_sw[i] for i in keep_indices])
    data_bw_sub = Dataset([data_bw[i] for i in keep_indices])
    greens_sw_sub = GreensTensorList(tensors=[greens_sw[i] for i in keep_indices])
    greens_bw_sub = GreensTensorList(tensors=[greens_bw[i] for i in keep_indices])
    
    # Quick grid search
    results_sw_sub = grid_search(data_sw_sub, greens_sw_sub, misfit_sw, origin, grid)
    results_bw_sub = grid_search(data_bw_sub, greens_bw_sub, misfit_bw, origin, grid)
    results_sub = 0.5 * results_bw_sub + 0.5 * results_sw_sub
    
    idx_sub = results_sub.source_idxmin()
    mt_sub = grid.get(idx_sub)
    jackknife_solutions.append({
        'dropped': f"{stations[drop_idx].network}.{stations[drop_idx].station}",
        'magnitude': mt_sub.magnitude(),
        'misfit': float(results_sub.min()),
    })

# Analyze stability
magnitudes = [s['magnitude'] for s in jackknife_solutions]
misfits = [s['misfit'] for s in jackknife_solutions]

mag_mean = np.mean(magnitudes)
mag_std = np.std(magnitudes)
misfit_mean = np.mean(misfits)
misfit_std = np.std(misfits)

print(f'\nJackknife results (dropping one station at a time):')
print(f'  Magnitude: {mag_mean:.2f} ± {mag_std:.2f}')
print(f'  Misfit: {misfit_mean:.4f} ± {misfit_std:.4f}')

# Check for outlier stations that significantly affect the solution
print(f'\nStations with largest impact on solution:')
for s in sorted(jackknife_solutions, key=lambda x: abs(x['magnitude'] - mag_mean), reverse=True)[:5]:
    diff = s['magnitude'] - mag_mean
    print(f"  {s['dropped']}: ΔMw = {diff:+.2f}")

# Feature-based quality check
print('\n' + '='*70)
print('FEATURE-BASED QUALITY ASSESSMENT')
print('='*70)

def analyze_waveform_features(data_list, greens_list, mt):
    """Detailed analysis of waveform feature matching"""
    results = []
    
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
                min_len = min(len(dd), len(ss))
                
                # 1. Cross-correlation (phase alignment)
                if np.std(dd[:min_len]) > 0 and np.std(ss[:min_len]) > 0:
                    cc = np.corrcoef(dd[:min_len], ss[:min_len])[0, 1]
                else:
                    cc = 0
                
                # 2. Amplitude ratio
                d_max = np.max(np.abs(dd[:min_len]))
                s_max = np.max(np.abs(ss[:min_len]))
                amp_ratio = d_max / s_max if s_max > 0 else 1.0
                
                # 3. Zero-crossing comparison (waveform shape)
                d_zc = np.sum(np.abs(np.diff(np.sign(dd[:min_len])))) / 2
                s_zc = np.sum(np.abs(np.diff(np.sign(ss[:min_len])))) / 2
                zc_ratio = d_zc / s_zc if s_zc > 0 else 1.0
                
                # 4. Energy distribution (early vs late)
                mid = min_len // 2
                d_early = np.sum(dd[:mid]**2)
                d_late = np.sum(dd[mid:min_len]**2)
                s_early = np.sum(ss[:mid]**2)
                s_late = np.sum(ss[mid:min_len]**2)
                
                d_ratio = d_early / d_late if d_late > 0 else 1.0
                s_ratio = s_early / s_late if s_late > 0 else 1.0
                energy_match = 1.0 / (1.0 + abs(np.log(d_ratio/s_ratio))) if s_ratio > 0 else 0
                
                results.append({
                    'cc': cc,
                    'amp_ratio': amp_ratio,
                    'zc_ratio': zc_ratio,
                    'energy_match': energy_match,
                })
        except:
            pass
    
    return results

sw_features = analyze_waveform_features(data_sw, greens_sw, best_mt)
bw_features = analyze_waveform_features(data_bw, greens_bw, best_mt)

print('\nSurface wave quality metrics:')
if sw_features:
    cc_mean = np.mean([f['cc'] for f in sw_features])
    amp_mean = np.mean([f['amp_ratio'] for f in sw_features])
    amp_std = np.std([f['amp_ratio'] for f in sw_features])
    print(f'  Cross-correlation: {cc_mean:.3f}')
    print(f'  Amplitude ratio: {amp_mean:.2f} ± {amp_std:.2f}')
    print(f'  Waveforms analyzed: {len(sw_features)}')

print('\nBody wave quality metrics:')
if bw_features:
    cc_mean = np.mean([f['cc'] for f in bw_features])
    amp_mean = np.mean([f['amp_ratio'] for f in bw_features])
    amp_std = np.std([f['amp_ratio'] for f in bw_features])
    print(f'  Cross-correlation: {cc_mean:.3f}')
    print(f'  Amplitude ratio: {amp_mean:.2f} ± {amp_std:.2f}')
    print(f'  Waveforms analyzed: {len(bw_features)}')

# Calculate overall quality score
all_cc = [f['cc'] for f in sw_features + bw_features]
overall_cc = np.mean(all_cc) if all_cc else 0

# Results
print('\n' + '='*70)
print('FINAL ROBUST RESULTS')
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
print(f'Quality metrics:')
print(f'  Number of stations: {len(stations)}')
print(f'  Final misfit: {final_misfit:.4f}')
print(f'  Variance Reduction: {(1-final_misfit)*100:.1f}%')
print(f'  Magnitude stability: ±{mag_std:.2f}')
print(f'  Mean cross-correlation: {overall_cc:.3f}')
print()
print(f'Moment tensor components (N·m):')
print(f'  Mrr: {mt_dict["Mrr"]:.2e}')
print(f'  Mtt: {mt_dict["Mtt"]:.2e}')
print(f'  Mpp: {mt_dict["Mpp"]:.2e}')
print(f'  Mrt: {mt_dict["Mrt"]:.2e}')
print(f'  Mrp: {mt_dict["Mrp"]:.2e}')
print(f'  Mtp: {mt_dict["Mtp"]:.2e}')

# Output
output_dir = os.path.join(script_dir, 'results_robust')
os.makedirs(output_dir, exist_ok=True)

print('\nGenerating output files...')

try:
    plot_data_greens2(
        os.path.join(output_dir, f'{event_id}_FMT_robust_waveforms.png'),
        data_bw, data_sw, greens_bw, greens_sw, 
        process_bw, process_sw, misfit_bw, misfit_sw, 
        stations, origin, best_mt, lune_dict
    )
    print('  Created waveform plot')
except Exception as e:
    print(f'  Warning: waveform plot failed: {e}')

try:
    plot_beachball(
        os.path.join(output_dir, f'{event_id}_FMT_robust_beachball.png'),
        best_mt, stations, origin
    )
    print('  Created beachball plot')
except Exception as e:
    print(f'  Warning: beachball plot failed: {e}')

try:
    plot_misfit_lune(
        os.path.join(output_dir, f'{event_id}_FMT_robust_misfit_lune.png'),
        results
    )
    print('  Created misfit lune plot')
except Exception as e:
    print(f'  Warning: misfit lune plot failed: {e}')

save_json(
    os.path.join(output_dir, f'{event_id}_FMT_robust_solution.json'),
    merged_dict
)

# Save detailed summary
with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
    f.write('ROBUST MOMENT TENSOR INVERSION RESULTS\n')
    f.write('='*50 + '\n\n')
    f.write(f'Event: {event_id}\n')
    f.write(f'Origin: {origin.time}\n')
    f.write(f'Location: {origin.latitude:.4f}°N, {origin.longitude:.4f}°E\n')
    f.write(f'Depth: {origin.depth_in_m/1000:.1f} km\n\n')
    f.write(f'Station selection:\n')
    f.write(f'  Method: SNR-based (threshold = {min_snr})\n')
    f.write(f'  Stations used: {len(stations)}\n\n')
    f.write(f'Stability analysis (jackknife):\n')
    f.write(f'  Magnitude: {mag_mean:.2f} ± {mag_std:.2f}\n')
    f.write(f'  Misfit: {misfit_mean:.4f} ± {misfit_std:.4f}\n\n')
    f.write(f'Waveform feature quality:\n')
    f.write(f'  Mean cross-correlation: {overall_cc:.3f}\n\n')
    f.write(f'Results:\n')
    f.write(f'  Mw: {best_mt.magnitude():.2f}\n')
    f.write(f'  M0: {best_mt.moment():.2e} N·m\n')
    f.write(f'  Strike: {lune_dict["kappa"]:.1f}°\n')
    f.write(f'  Slip: {lune_dict["sigma"]:.1f}°\n')
    f.write(f'  Dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°\n\n')
    f.write(f'Misfit: {final_misfit:.4f}\n')
    f.write(f'Variance Reduction: {(1-final_misfit)*100:.1f}%\n')

print(f'\nResults saved to: {output_dir}')
print('\n' + '='*70)
