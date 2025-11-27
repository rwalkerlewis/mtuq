#!/usr/bin/env python
"""
Detailed station quality analysis to identify best-fitting stations
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/workspace')

from mtuq import read, download_greens
from mtuq.event import Origin, MomentTensor
from mtuq.misfit import Misfit
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
magnitude = 6.1

script_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')
weights_filtered = os.path.join(script_dir, 'weights_filtered.dat')

# Use original best MT from previous run
# Mrr, Mtt, Mpp, Mrt, Mrp, Mtp in USE convention
mt_array = np.array([-1.07e+18, -1.00e+18, 1.49e+18, -9.14e+16, 6.79e+17, 7.14e+17])
best_mt = MomentTensor(mt_array, convention='USE')

print('='*70)
print('STATION QUALITY ANALYSIS')
print('='*70)

# Process parameters - same as original inversion
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

wavelet = Trapezoid(magnitude=magnitude)
greens.convolve(wavelet)
greens_bw = greens.map(process_bw)
greens_sw = greens.map(process_sw)

# Analyze each station
print('\nAnalyzing station fits...')
station_quality = []

for i, station in enumerate(stations):
    sta_id = f"{station.network}.{station.station}."
    
    # Calculate distance from coordinates
    from obspy.geodetics import gps2dist_azimuth
    dist_m, az, _ = gps2dist_azimuth(
        origin.latitude, origin.longitude,
        station.latitude, station.longitude
    )
    dist_km = dist_m / 1000.0
    
    result = {
        'station_id': sta_id,
        'network': station.network,
        'station': station.station,
        'distance_km': dist_km,
        'azimuth': az,
        'sw_vr': None,
        'bw_vr': None,
        'sw_cc': None,
        'bw_cc': None,
        'combined_vr': None,
    }
    
    # Surface wave fit
    try:
        if i < len(data_sw) and data_sw[i] is not None and len(data_sw[i]) > 0:
            d = data_sw[i]
            g = greens_sw[i]
            components = [tr.stats.channel[-1] for tr in d]
            g._set_components(components)
            s = g.get_synthetics(best_mt, components)
            
            d_energy = 0
            res_energy = 0
            cc_total = 0
            n_comp = 0
            
            for j, (tr_d, tr_s) in enumerate(zip(d, s)):
                dd = tr_d.data
                ss = tr_s.data
                min_len = min(len(dd), len(ss))
                
                # Variance reduction
                d_energy += np.sum(dd[:min_len]**2)
                res_energy += np.sum((dd[:min_len] - ss[:min_len])**2)
                
                # Cross-correlation
                if np.std(dd[:min_len]) > 0 and np.std(ss[:min_len]) > 0:
                    cc = np.corrcoef(dd[:min_len], ss[:min_len])[0, 1]
                    cc_total += cc
                    n_comp += 1
            
            if d_energy > 0:
                result['sw_vr'] = 1.0 - res_energy / d_energy
            if n_comp > 0:
                result['sw_cc'] = cc_total / n_comp
    except Exception as e:
        pass
    
    # Body wave fit
    try:
        if i < len(data_bw) and data_bw[i] is not None and len(data_bw[i]) > 0:
            d = data_bw[i]
            g = greens_bw[i]
            components = [tr.stats.channel[-1] for tr in d]
            g._set_components(components)
            s = g.get_synthetics(best_mt, components)
            
            d_energy = 0
            res_energy = 0
            cc_total = 0
            n_comp = 0
            
            for j, (tr_d, tr_s) in enumerate(zip(d, s)):
                dd = tr_d.data
                ss = tr_s.data
                min_len = min(len(dd), len(ss))
                
                d_energy += np.sum(dd[:min_len]**2)
                res_energy += np.sum((dd[:min_len] - ss[:min_len])**2)
                
                if np.std(dd[:min_len]) > 0 and np.std(ss[:min_len]) > 0:
                    cc = np.corrcoef(dd[:min_len], ss[:min_len])[0, 1]
                    cc_total += cc
                    n_comp += 1
            
            if d_energy > 0:
                result['bw_vr'] = 1.0 - res_energy / d_energy
            if n_comp > 0:
                result['bw_cc'] = cc_total / n_comp
    except Exception as e:
        pass
    
    # Combined VR (weight SW more for deep earthquake)
    vrs = []
    if result['sw_vr'] is not None:
        vrs.append(('sw', result['sw_vr'], 0.67))
    if result['bw_vr'] is not None:
        vrs.append(('bw', result['bw_vr'], 0.33))
    
    if vrs:
        result['combined_vr'] = sum(v * w for _, v, w in vrs) / sum(w for _, _, w in vrs)
    
    station_quality.append(result)

# Sort by combined VR
station_quality.sort(key=lambda x: x['combined_vr'] if x['combined_vr'] is not None else -999, reverse=True)

# Print results
print('\n' + '='*90)
print(f"{'Station':<15} {'Dist (km)':<10} {'Az':<8} {'SW VR':<10} {'BW VR':<10} {'SW CC':<8} {'BW CC':<8} {'Combined':<10}")
print('='*90)

for r in station_quality:
    sw_vr = f"{r['sw_vr']*100:.1f}%" if r['sw_vr'] is not None else "N/A"
    bw_vr = f"{r['bw_vr']*100:.1f}%" if r['bw_vr'] is not None else "N/A"
    sw_cc = f"{r['sw_cc']:.2f}" if r['sw_cc'] is not None else "N/A"
    bw_cc = f"{r['bw_cc']:.2f}" if r['bw_cc'] is not None else "N/A"
    combined = f"{r['combined_vr']*100:.1f}%" if r['combined_vr'] is not None else "N/A"
    
    print(f"{r['station_id']:<15} {r['distance_km']:<10.1f} {r['azimuth']:<8.1f} {sw_vr:<10} {bw_vr:<10} {sw_cc:<8} {bw_cc:<8} {combined:<10}")

# Identify best stations (VR > 0 or CC > 0.3)
print('\n' + '='*70)
print('STATION SELECTION')
print('='*70)

# Strategy: Select stations with positive VR or good CC
best_stations = []
marginal_stations = []
poor_stations = []

for r in station_quality:
    combined_vr = r['combined_vr']
    sw_cc = r['sw_cc'] if r['sw_cc'] is not None else 0
    bw_cc = r['bw_cc'] if r['bw_cc'] is not None else 0
    avg_cc = (sw_cc + bw_cc) / 2 if sw_cc != 0 and bw_cc != 0 else max(sw_cc, bw_cc)
    
    if combined_vr is not None and combined_vr > 0.1:
        best_stations.append(r)
    elif combined_vr is not None and combined_vr > -0.5 and avg_cc > 0.3:
        marginal_stations.append(r)
    else:
        poor_stations.append(r)

print(f'\nBest stations (VR > 10%): {len(best_stations)}')
for r in best_stations:
    print(f"  {r['station_id']} (VR={r['combined_vr']*100:.1f}%)")

print(f'\nMarginal stations (VR > -50%, CC > 0.3): {len(marginal_stations)}')
for r in marginal_stations:
    sw_cc = r['sw_cc'] if r['sw_cc'] is not None else 0
    bw_cc = r['bw_cc'] if r['bw_cc'] is not None else 0
    print(f"  {r['station_id']} (VR={r['combined_vr']*100:.1f}%, CC_sw={sw_cc:.2f}, CC_bw={bw_cc:.2f})")

print(f'\nPoor stations (excluded): {len(poor_stations)}')
for r in poor_stations:
    print(f"  {r['station_id']}")

# Create highly selective weights file
weights_best = os.path.join(script_dir, 'weights_best.dat')
best_ids = [r['station_id'] for r in best_stations]

with open(weights_filtered, 'r') as f:
    lines = f.readlines()

with open(weights_best, 'w') as f:
    f.write('# Best-fitting stations only (VR > 10%)\n')
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) > 0:
            # Extract network.station. from line
            sta_code = parts[0]
            network = sta_code.split('.')[1]
            station = sta_code.split('.')[2]
            sta_id = f"{network}.{station}."
            if sta_id in best_ids:
                f.write(line)

print(f'\nCreated weights file with {len(best_stations)} best stations: {weights_best}')

# Save analysis
with open(os.path.join(script_dir, 'station_quality_report.txt'), 'w') as f:
    f.write('STATION QUALITY ANALYSIS REPORT\n')
    f.write('='*70 + '\n\n')
    f.write(f"{'Station':<15} {'Dist':<10} {'SW VR':<12} {'BW VR':<12} {'Combined':<12}\n")
    f.write('-'*70 + '\n')
    for r in station_quality:
        sw_vr = f"{r['sw_vr']*100:.1f}%" if r['sw_vr'] is not None else "N/A"
        bw_vr = f"{r['bw_vr']*100:.1f}%" if r['bw_vr'] is not None else "N/A"
        combined = f"{r['combined_vr']*100:.1f}%" if r['combined_vr'] is not None else "N/A"
        f.write(f"{r['station_id']:<15} {r['distance_km']:<10.1f} {sw_vr:<12} {bw_vr:<12} {combined:<12}\n")
    f.write('\n\nBest stations:\n')
    for r in best_stations:
        f.write(f"  {r['station_id']}\n")

print('\nStation quality report saved.')
