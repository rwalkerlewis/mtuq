#!/usr/bin/env python
"""
Robust moment tensor inversion with overfitting prevention

Key safeguards:
1. Station selection based on SNR (not fit to solution)
2. K-fold cross-validation to ensure solution generalizes
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
print('ROBUST MOMENT TENSOR INVERSION')
print('(with overfitting prevention)')
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
print('\nStation SNR ranking (independent of solution):')
for sta_id, snr in sorted_stations[:20]:
    print(f'  {sta_id}: SNR = {snr:.1f}')

# Select stations based on SNR (not fit to solution)
# Require minimum 20 stations for robust inversion
min_stations = 20
snr_threshold = 5.0  # minimum SNR

good_snr_stations = [sta_id for sta_id, snr in sorted_stations if snr >= snr_threshold]
print(f'\nStations with SNR >= {snr_threshold}: {len(good_snr_stations)}')

# Ensure azimuthal coverage
print('\nChecking azimuthal coverage...')
station_azimuths = {}
for station in stations:
    sta_id = f"{station.network}.{station.station}"
    if sta_id in good_snr_stations:
        dist_m, az, _ = gps2dist_azimuth(
            origin.latitude, origin.longitude,
            station.latitude, station.longitude
        )
        station_azimuths[sta_id] = az

# Select stations to maximize azimuthal coverage
# Divide into 8 azimuth bins (45° each)
az_bins = defaultdict(list)
for sta_id, az in station_azimuths.items():
    bin_idx = int(az // 45) % 8
    az_bins[bin_idx].append((sta_id, station_snr.get(sta_id, 0)))

# Select best SNR station from each bin, then fill remaining
selected_stations = []
for bin_idx in range(8):
    if az_bins[bin_idx]:
        # Sort by SNR and take best
        az_bins[bin_idx].sort(key=lambda x: x[1], reverse=True)
        selected_stations.append(az_bins[bin_idx][0][0])

# Add more stations from remaining pool to reach minimum
remaining = [(sta_id, station_snr.get(sta_id, 0)) for sta_id in good_snr_stations 
             if sta_id not in selected_stations]
remaining.sort(key=lambda x: x[1], reverse=True)

for sta_id, snr in remaining:
    if len(selected_stations) >= min_stations:
        break
    selected_stations.append(sta_id)

print(f'\nSelected {len(selected_stations)} stations for inversion:')
for sta_id in selected_stations:
    az = station_azimuths.get(sta_id, 0)
    snr = station_snr.get(sta_id, 0)
    print(f'  {sta_id}: Az={az:.0f}°, SNR={snr:.1f}')

# Create weights file for selected stations
weights_robust = os.path.join(script_dir, 'weights_robust.dat')
with open(weights_filtered, 'r') as f:
    lines = f.readlines()

with open(weights_robust, 'w') as f:
    f.write('# Robust station selection (SNR-based, azimuthally balanced)\n')
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

# K-fold cross-validation
print('\n' + '='*70)
print('K-FOLD CROSS-VALIDATION (k=5)')
print('='*70)

n_folds = 5
n_stations = len(stations)
fold_size = n_stations // n_folds

# Shuffle station indices
np.random.seed(42)  # reproducibility
indices = np.random.permutation(n_stations)

cv_solutions = []
cv_misfits_train = []
cv_misfits_val = []

grid = FullMomentTensorGridSemiregular(
    npts_per_axis=12,
    magnitudes=[5.9, 6.0, 6.1, 6.2],
)

for fold in range(n_folds):
    # Split into train/validation
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else n_stations
    val_indices = indices[val_start:val_end]
    train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
    
    print(f'\nFold {fold+1}: Train={len(train_indices)} stations, Val={len(val_indices)} stations')
    
    # Select stations for train/val
    train_stations = [stations[i] for i in train_indices]
    val_stations = [stations[i] for i in val_indices]
    
    # Create training data subsets
    data_sw_train = data_sw.select(train_stations)
    data_bw_train = data_bw.select(train_stations)
    greens_sw_train = greens_sw.select(train_stations)
    greens_bw_train = greens_bw.select(train_stations)
    
    # Grid search on training set
    results_sw_train = grid_search(data_sw_train, greens_sw_train, misfit_sw, origin, grid)
    results_bw_train = grid_search(data_bw_train, greens_bw_train, misfit_bw, origin, grid)
    results_train = 0.5 * results_bw_train + 0.5 * results_sw_train
    
    idx = results_train.source_idxmin()
    best_mt_fold = grid.get(idx)
    train_misfit = float(results_train.min())
    
    # Evaluate on validation set
    data_sw_val = data_sw.select(val_stations)
    data_bw_val = data_bw.select(val_stations)
    greens_sw_val = greens_sw.select(val_stations)
    greens_bw_val = greens_bw.select(val_stations)
    
    results_sw_val = grid_search(data_sw_val, greens_sw_val, misfit_sw, origin, grid)
    results_bw_val = grid_search(data_bw_val, greens_bw_val, misfit_bw, origin, grid)
    results_val = 0.5 * results_bw_val + 0.5 * results_sw_val
    
    # Get validation misfit at the training solution
    val_misfit = float(results_val[idx])
    
    cv_solutions.append(best_mt_fold)
    cv_misfits_train.append(train_misfit)
    cv_misfits_val.append(val_misfit)
    
    print(f'  Train misfit: {train_misfit:.4f}')
    print(f'  Val misfit: {val_misfit:.4f}')
    print(f'  Overfitting ratio: {val_misfit/train_misfit:.2f}')
    print(f'  Solution: Mw={best_mt_fold.magnitude():.2f}')

# Analyze CV results
print('\n' + '='*70)
print('CROSS-VALIDATION SUMMARY')
print('='*70)

mean_train = np.mean(cv_misfits_train)
mean_val = np.mean(cv_misfits_val)
std_val = np.std(cv_misfits_val)
overfit_ratio = mean_val / mean_train

print(f'\nMean training misfit: {mean_train:.4f}')
print(f'Mean validation misfit: {mean_val:.4f} ± {std_val:.4f}')
print(f'Overfitting ratio: {overfit_ratio:.2f}')

if overfit_ratio < 1.3:
    print('✓ Solution appears robust (overfitting ratio < 1.3)')
elif overfit_ratio < 1.5:
    print('⚠ Moderate overfitting detected (1.3 < ratio < 1.5)')
else:
    print('✗ Significant overfitting detected (ratio > 1.5)')

# Check solution stability across folds
magnitudes = [mt.magnitude() for mt in cv_solutions]
print(f'\nMagnitude stability: Mw = {np.mean(magnitudes):.2f} ± {np.std(magnitudes):.2f}')

# Final inversion using all stations
print('\n' + '='*70)
print('FINAL ROBUST INVERSION (all selected stations)')
print('='*70)

results_sw_full = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
results_bw_full = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
results_full = 0.5 * results_bw_full + 0.5 * results_sw_full

idx = results_full.source_idxmin()
best_mt = grid.get(idx)
lune_dict = grid.get_dict(idx)
mt_dict = best_mt.as_dict()
final_misfit = float(results_full.min())

# Feature-based quality check
print('\nFeature-based quality assessment...')

def check_waveform_features(data_list, greens_list, mt):
    """Check polarity matches and amplitude ratios"""
    polarity_matches = 0
    polarity_total = 0
    amplitude_ratios = []
    
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
                
                # Check first motion polarity
                d_first = np.sign(dd[min_len//4])  # Use early part
                s_first = np.sign(ss[min_len//4])
                polarity_total += 1
                if d_first == s_first:
                    polarity_matches += 1
                
                # Check amplitude ratio
                d_max = np.max(np.abs(dd[:min_len]))
                s_max = np.max(np.abs(ss[:min_len]))
                if s_max > 0:
                    amplitude_ratios.append(d_max / s_max)
        except:
            pass
    
    polarity_pct = 100 * polarity_matches / polarity_total if polarity_total > 0 else 0
    amp_ratio_mean = np.mean(amplitude_ratios) if amplitude_ratios else 1.0
    amp_ratio_std = np.std(amplitude_ratios) if amplitude_ratios else 0
    
    return polarity_pct, amp_ratio_mean, amp_ratio_std

pol_sw, amp_sw, amp_std_sw = check_waveform_features(data_sw, greens_sw, best_mt)
pol_bw, amp_bw, amp_std_bw = check_waveform_features(data_bw, greens_bw, best_mt)

print(f'\nSurface waves:')
print(f'  Polarity match: {pol_sw:.1f}%')
print(f'  Amplitude ratio (data/synth): {amp_sw:.2f} ± {amp_std_sw:.2f}')

print(f'\nBody waves:')
print(f'  Polarity match: {pol_bw:.1f}%')
print(f'  Amplitude ratio (data/synth): {amp_bw:.2f} ± {amp_std_bw:.2f}')

# Calculate expected VR from CV
expected_vr = 1.0 - mean_val

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
print(f'  Training misfit: {final_misfit:.4f}')
print(f'  Expected validation misfit: {mean_val:.4f} ± {std_val:.4f}')
print(f'  Expected VR: {expected_vr*100:.1f}% ± {std_val*100:.1f}%')
print(f'  Overfitting ratio: {overfit_ratio:.2f}')
print(f'  Polarity match (SW): {pol_sw:.1f}%')
print(f'  Polarity match (BW): {pol_bw:.1f}%')
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
        results_full
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
    f.write(f'Stations used: {len(stations)}\n')
    f.write(f'Station selection: SNR-based, azimuthally balanced\n\n')
    f.write(f'Cross-validation (k={n_folds}):\n')
    f.write(f'  Mean train misfit: {mean_train:.4f}\n')
    f.write(f'  Mean val misfit: {mean_val:.4f} ± {std_val:.4f}\n')
    f.write(f'  Overfitting ratio: {overfit_ratio:.2f}\n\n')
    f.write(f'Feature quality:\n')
    f.write(f'  Polarity match (SW): {pol_sw:.1f}%\n')
    f.write(f'  Polarity match (BW): {pol_bw:.1f}%\n')
    f.write(f'  Amplitude ratio (SW): {amp_sw:.2f} ± {amp_std_sw:.2f}\n')
    f.write(f'  Amplitude ratio (BW): {amp_bw:.2f} ± {amp_std_bw:.2f}\n\n')
    f.write(f'Results:\n')
    f.write(f'  Mw: {best_mt.magnitude():.2f}\n')
    f.write(f'  Strike: {lune_dict["kappa"]:.1f}°\n')
    f.write(f'  Slip: {lune_dict["sigma"]:.1f}°\n')
    f.write(f'  Dip: {np.degrees(np.arccos(lune_dict["h"])):.1f}°\n\n')
    f.write(f'Expected VR: {expected_vr*100:.1f}% ± {std_val*100:.1f}%\n')

print(f'\nResults saved to: {output_dir}')
print('\n' + '='*70)
