#!/usr/bin/env python
"""
Analyze waveform fits using MTUQ's misfit calculation per station
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
from mtuq.misfit.waveform import level0, level1, level2
from mtuq.process_data import ProcessData
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

# Misfit functions - without normalization to get raw values
misfit_sw = Misfit(
    norm='L2',
    time_shift_min=-15.,
    time_shift_max=+15.,
    time_shift_groups=['ZR', 'T'],
    normalize=False,  # Get raw misfit values
)

misfit_bw = Misfit(
    norm='L2',
    time_shift_min=-5.,
    time_shift_max=+5.,
    time_shift_groups=['ZR'],
    normalize=False,
)

print('Reading data...')
station_id_list = list(parse_station_codes(path_weights))
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

# Use the best-fitting MT from previous run
from mtuq.event import MomentTensor
import numpy as np
# Array format: [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp] in USE convention
mt_array = np.array([-1.07e+18, -1.00e+18, 1.49e+18, -9.14e+16, 6.79e+17, 7.14e+17])
best_mt = MomentTensor(mt_array, convention='USE')

print('\nCalculating per-station misfits...')

station_misfits = []

for i, station in enumerate(stations):
    station_id = station.id
    
    try:
        # Calculate variance reduction for this station
        sw_vr = None
        bw_vr = None
        
        # Surface wave
        if i < len(data_sw) and data_sw[i] is not None and len(data_sw[i]) > 0:
            d_sw = data_sw[i]
            g_sw = greens_sw[i]
            
            # Get synthetics
            components = [tr.stats.channel[-1] for tr in d_sw]
            g_sw._set_components(components)
            s_sw = g_sw.get_synthetics(best_mt)
            
            # Calculate normalized misfit (variance reduction)
            d_energy = 0
            residual_energy = 0
            for j, tr in enumerate(d_sw):
                d = tr.data
                if j < len(s_sw):
                    s = s_sw[j].data
                    min_len = min(len(d), len(s))
                    d_energy += np.sum(d[:min_len]**2)
                    residual_energy += np.sum((d[:min_len] - s[:min_len])**2)
            
            if d_energy > 0:
                sw_vr = 1.0 - residual_energy / d_energy
        
        # Body wave
        if i < len(data_bw) and data_bw[i] is not None and len(data_bw[i]) > 0:
            d_bw = data_bw[i]
            g_bw = greens_bw[i]
            
            components = [tr.stats.channel[-1] for tr in d_bw]
            g_bw._set_components(components)
            s_bw = g_bw.get_synthetics(best_mt)
            
            d_energy = 0
            residual_energy = 0
            for j, tr in enumerate(d_bw):
                d = tr.data
                if j < len(s_bw):
                    s = s_bw[j].data
                    min_len = min(len(d), len(s))
                    d_energy += np.sum(d[:min_len]**2)
                    residual_energy += np.sum((d[:min_len] - s[:min_len])**2)
            
            if d_energy > 0:
                bw_vr = 1.0 - residual_energy / d_energy
        
        # Combined variance reduction (weighted)
        if sw_vr is not None and bw_vr is not None:
            combined_vr = 0.7 * sw_vr + 0.3 * bw_vr
        elif sw_vr is not None:
            combined_vr = sw_vr
        elif bw_vr is not None:
            combined_vr = bw_vr
        else:
            combined_vr = None
        
        if combined_vr is not None:
            station_misfits.append({
                'id': station_id,
                'vr': combined_vr,
                'sw_vr': sw_vr,
                'bw_vr': bw_vr,
                'lat': station.latitude,
                'lon': station.longitude,
            })
    
    except Exception as e:
        print(f'  Error for {station_id}: {e}')

# Sort by variance reduction (lowest = worst fit)
station_misfits.sort(key=lambda x: x['vr'])

print('\n' + '='*70)
print('STATION VARIANCE REDUCTION (sorted by worst fit)')
print('='*70)
print(f'{"Station":<20} {"VR Total":>10} {"VR SW":>10} {"VR BW":>10} {"Status":>10}')
print('-'*70)

# Calculate threshold for removing stations
vr_values = [s['vr'] for s in station_misfits]
mean_vr = np.mean(vr_values)
std_vr = np.std(vr_values)
threshold = mean_vr - 1.0 * std_vr  # Remove stations more than 1 std below mean

print(f'\nStatistics:')
print(f'  Mean VR: {mean_vr:.4f}')
print(f'  Std VR: {std_vr:.4f}')
print(f'  Threshold (mean - 1*std): {threshold:.4f}')
print(f'  VR < 0 indicates poor fit (synthetics worse than mean)')
print()

stations_to_remove = []
stations_to_keep = []

for s in station_misfits:
    sw_str = f'{s["sw_vr"]:.4f}' if s['sw_vr'] is not None else 'N/A'
    bw_str = f'{s["bw_vr"]:.4f}' if s['bw_vr'] is not None else 'N/A'
    
    # Remove if VR is below threshold or negative
    if s['vr'] < threshold or s['vr'] < 0:
        status = 'REMOVE'
        stations_to_remove.append(s['id'])
    else:
        status = 'keep'
        stations_to_keep.append(s['id'])
    
    print(f'{s["id"]:<20} {s["vr"]:>10.4f} {sw_str:>10} {bw_str:>10} {status:>10}')

print('\n' + '='*70)
print(f'SUMMARY: Remove {len(stations_to_remove)} stations, keep {len(stations_to_keep)} stations')
print('='*70)

print(f'\nStations to REMOVE ({len(stations_to_remove)}):')
for s in stations_to_remove:
    print(f'  {s}')

# Save stations to remove
with open(os.path.join(script_dir, 'stations_to_remove.txt'), 'w') as f:
    f.write(f'# Stations with poor fits (VR < {threshold:.4f})\n')
    for s in stations_to_remove:
        f.write(f'{s}\n')

# Create updated weights file
print('\nCreating updated weights file...')
with open(path_weights, 'r') as f:
    lines = f.readlines()

with open(os.path.join(script_dir, 'weights_filtered.dat'), 'w') as f:
    for line in lines:
        if line.startswith('#'):
            f.write(line)
            continue
        
        # Check if this station should be kept
        keep = True
        for remove_id in stations_to_remove:
            if remove_id.replace('.', '') in line.replace('.', '') or remove_id in line:
                # More careful matching
                parts = line.split()
                if len(parts) > 0:
                    line_station = '.'.join(parts[0].split('.')[1:4])
                    if line_station == remove_id:
                        keep = False
                        break
        
        if keep:
            f.write(line)

print(f'Saved filtered weights to: weights_filtered.dat')
print(f'Remaining stations: {len(stations_to_keep)}')
