#!/usr/bin/env python
"""
Analyze waveform fits and identify stations with poor fits
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/workspace')

from mtuq import read, download_greens
from mtuq.event import Origin
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath
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

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')
path_weights = os.path.join(script_dir, 'weights.dat')

# Processing parameters
process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.02,
    freq_max=0.1,
    pick_type='user_supplied',
    window_type='surface_wave',
    window_length=150.,
    capuaf_file=path_weights,
)

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.03,
    freq_max=0.15,
    pick_type='user_supplied',
    window_type='body_wave',
    window_length=30.,
    capuaf_file=path_weights,
)

# Misfit functions
misfit_sw = Misfit(
    norm='L2',
    time_shift_min=-15.,
    time_shift_max=+15.,
    time_shift_groups=['ZR', 'T'],
    normalize=True,
)

misfit_bw = Misfit(
    norm='L2',
    time_shift_min=-5.,
    time_shift_max=+5.,
    time_shift_groups=['ZR'],
    normalize=True,
)

print('Reading data...')
station_id_list = parse_station_codes(path_weights)
data = read(path_data, format='sac',
    event_id=event_id,
    station_id_list=station_id_list,
    tags=['units:m', 'type:velocity'])

data.sort_by_distance()
stations = data.get_stations()

print('Processing data...')
data_bw = data.map(process_bw)
data_sw = data.map(process_sw)

print('Downloading Green\'s functions...')
greens = download_greens(stations, origin, model)

wavelet = Trapezoid(magnitude=magnitude)
greens.convolve(wavelet)
greens_bw = greens.map(process_bw)
greens_sw = greens.map(process_sw)

# Use a smaller grid for quick analysis
grid = FullMomentTensorGridSemiregular(
    npts_per_axis=10,
    magnitudes=[6.0, 6.1, 6.2],
)

print('Running grid search...')
results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
results = 0.3 * results_bw + 0.7 * results_sw

# Get best-fitting moment tensor
idx = results.source_idxmin()
best_mt = grid.get(idx)

print('\nAnalyzing station fits...')

# Calculate per-station misfit
station_misfits = []

for i, station in enumerate(stations):
    station_id = station.id
    
    # Get data and synthetics for this station
    try:
        # Surface wave misfit
        sw_misfit = 0
        sw_count = 0
        if i < len(data_sw) and data_sw[i] is not None:
            for trace in data_sw[i]:
                if hasattr(trace, 'weight') and trace.weight > 0:
                    d = trace.data
                    # Get synthetic
                    greens_sw[i]._set_components([trace.stats.channel[-1]])
                    s = greens_sw[i].get_synthetics(best_mt)
                    if len(s) > 0:
                        s_data = s[0].data
                        # Normalize and compute misfit
                        d_norm = np.linalg.norm(d)
                        if d_norm > 0:
                            misfit_val = np.linalg.norm(d - s_data[:len(d)]) / d_norm
                            sw_misfit += misfit_val
                            sw_count += 1
        
        # Body wave misfit
        bw_misfit = 0
        bw_count = 0
        if i < len(data_bw) and data_bw[i] is not None:
            for trace in data_bw[i]:
                if hasattr(trace, 'weight') and trace.weight > 0:
                    d = trace.data
                    greens_bw[i]._set_components([trace.stats.channel[-1]])
                    s = greens_bw[i].get_synthetics(best_mt)
                    if len(s) > 0:
                        s_data = s[0].data
                        d_norm = np.linalg.norm(d)
                        if d_norm > 0:
                            misfit_val = np.linalg.norm(d - s_data[:len(d)]) / d_norm
                            bw_misfit += misfit_val
                            bw_count += 1
        
        # Combined misfit
        total_misfit = 0
        total_count = 0
        if sw_count > 0:
            total_misfit += 0.7 * sw_misfit / sw_count
            total_count += 1
        if bw_count > 0:
            total_misfit += 0.3 * bw_misfit / bw_count
            total_count += 1
        
        if total_count > 0:
            avg_misfit = total_misfit / total_count
            station_misfits.append((station_id, avg_misfit, station.latitude, station.longitude))
    
    except Exception as e:
        print(f'  Error for {station_id}: {e}')

# Sort by misfit
station_misfits.sort(key=lambda x: x[1], reverse=True)

print('\n' + '='*60)
print('STATION MISFITS (sorted by worst fit)')
print('='*60)

# Calculate statistics
misfits_only = [m[1] for m in station_misfits]
mean_misfit = np.mean(misfits_only)
std_misfit = np.std(misfits_only)
threshold = mean_misfit + 1.5 * std_misfit

print(f'\nMean misfit: {mean_misfit:.4f}')
print(f'Std dev: {std_misfit:.4f}')
print(f'Threshold (mean + 1.5*std): {threshold:.4f}')
print()

# Identify stations to remove
stations_to_remove = []
stations_to_keep = []

for station_id, misfit, lat, lon in station_misfits:
    status = 'REMOVE' if misfit > threshold else 'keep'
    print(f'{station_id:20s}  misfit: {misfit:.4f}  {status}')
    if misfit > threshold:
        stations_to_remove.append(station_id)
    else:
        stations_to_keep.append(station_id)

print(f'\n{len(stations_to_remove)} stations to remove:')
for s in stations_to_remove:
    print(f'  {s}')

print(f'\n{len(stations_to_keep)} stations to keep')

# Save list of stations to remove
with open(os.path.join(script_dir, 'stations_to_remove.txt'), 'w') as f:
    f.write('# Stations with poor fits (misfit > mean + 1.5*std)\n')
    f.write(f'# Threshold: {threshold:.4f}\n')
    for s in stations_to_remove:
        f.write(f'{s}\n')

print(f'\nSaved stations to remove to: stations_to_remove.txt')
