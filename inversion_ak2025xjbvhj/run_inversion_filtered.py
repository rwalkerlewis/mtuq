#!/usr/bin/env python
"""
Full Moment Tensor Inversion with filtered stations (poor fits removed)
"""

import os
import sys
import numpy as np

sys.path.insert(0, '/workspace')

from mtuq import read, download_greens
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid


if __name__ == '__main__':
    
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
    
    # File paths - USE FILTERED WEIGHTS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')
    path_weights = os.path.join(script_dir, 'weights_filtered.dat')  # FILTERED!
    output_dir = os.path.join(script_dir, 'results_filtered')
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Station selection from filtered weights
    print('='*60)
    print('INVERSION WITH FILTERED STATIONS (poor fits removed)')
    print('='*60)
    print()
    
    print('Parsing station codes from FILTERED weights file...\n')
    station_id_list = parse_station_codes(path_weights)
    print(f'Using {len(list(station_id_list))} stations (16 poor-fit stations removed)\n')
    
    # Moment tensor grid
    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=10,
        magnitudes=[5.8, 5.9, 6.0, 6.1, 6.2],
    )
    
    print(f'Grid size: {grid.size} moment tensors\n')
    
    wavelet = Trapezoid(magnitude=magnitude)
    
    # Read and process data
    print('Reading waveform data...\n')
    station_id_list = parse_station_codes(path_weights)
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:m', 'type:velocity'])
    
    print(f'Read {len(data)} stations\n')
    
    data.sort_by_distance()
    stations = data.get_stations()
    
    print('Processing data...\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)
    
    # Download Green's functions
    print('Downloading Green\'s functions from Syngine...\n')
    greens = download_greens(stations, origin, model)
    
    print('Processing Green\'s functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)
    
    # Grid search
    print('='*60)
    print('Starting grid search...')
    print('='*60 + '\n')
    
    print('Evaluating surface wave misfit...\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
    
    print('Evaluating body wave misfit...\n')
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
    
    results = 0.3 * results_bw + 0.7 * results_sw
    
    # Extract best-fitting source
    print('\n' + '='*60)
    print('RESULTS (FILTERED STATIONS)')
    print('='*60 + '\n')
    
    idx = results.source_idxmin()
    best_mt = grid.get(idx)
    lune_dict = grid.get_dict(idx)
    mt_dict = best_mt.as_dict()
    
    merged_dict = merge_dicts(
        mt_dict, lune_dict, 
        {'M0': best_mt.moment()},
        {'Mw': best_mt.magnitude()}, 
        origin
    )
    
    print('Best-fitting moment tensor:')
    print(f'  Magnitude (Mw): {best_mt.magnitude():.2f}')
    print(f'  Moment (M0): {best_mt.moment():.2e} N·m')
    print()
    print('Lune/Source coordinates:')
    print(f'  v: {lune_dict["v"]:.3f}')
    print(f'  w: {lune_dict["w"]:.3f}')
    print()
    print('Fault parameters:')
    print(f'  kappa (strike): {lune_dict["kappa"]:.1f}°')
    print(f'  sigma (slip): {lune_dict["sigma"]:.1f}°')
    print(f'  h (cos dip): {lune_dict["h"]:.3f}')
    print()
    print('Moment tensor components (in N·m):')
    print(f'  Mrr: {mt_dict["Mrr"]:.2e}')
    print(f'  Mtt: {mt_dict["Mtt"]:.2e}')
    print(f'  Mpp: {mt_dict["Mpp"]:.2e}')
    print(f'  Mrt: {mt_dict["Mrt"]:.2e}')
    print(f'  Mrp: {mt_dict["Mrp"]:.2e}')
    print(f'  Mtp: {mt_dict["Mtp"]:.2e}')
    print()
    
    # Generate output files
    print('Generating output files...\n')
    
    try:
        plot_data_greens2(
            os.path.join(output_dir, f'{event_id}_FMT_filtered_waveforms.png'),
            data_bw, data_sw, greens_bw, greens_sw, 
            process_bw, process_sw,
            misfit_bw, misfit_sw, 
            stations, origin, best_mt, lune_dict
        )
        print(f'  Created waveform comparison plot')
    except Exception as e:
        print(f'  Warning: Could not create waveform plot: {e}')
    
    try:
        plot_beachball(
            os.path.join(output_dir, f'{event_id}_FMT_filtered_beachball.png'),
            best_mt, stations, origin
        )
        print(f'  Created beachball plot')
    except Exception as e:
        print(f'  Warning: Could not create beachball plot: {e}')
    
    try:
        plot_misfit_lune(
            os.path.join(output_dir, f'{event_id}_FMT_filtered_misfit_lune.png'),
            results
        )
        print(f'  Created misfit lune plot')
    except Exception as e:
        print(f'  Warning: Could not create misfit lune plot: {e}')
    
    save_json(
        os.path.join(output_dir, f'{event_id}_FMT_filtered_solution.json'),
        merged_dict
    )
    print(f'  Saved solution to {event_id}_FMT_filtered_solution.json')
    
    print('\n' + '='*60)
    print('INVERSION COMPLETE (FILTERED)')
    print('='*60)
    print(f'\nResults saved to: {output_dir}')
    print(f'Stations used: {len(stations)} (16 poor-fit stations removed)')
