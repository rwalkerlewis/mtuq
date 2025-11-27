#!/usr/bin/env python
"""
Full Moment Tensor Inversion for Alaska Earthquake ak2025xjbvhj
M6.0 earthquake near Susitna, Alaska on 2025-11-27

This script performs a grid search over all six independent moment tensor
parameters to find the best-fitting source mechanism.
"""

import os
import sys
import numpy as np

# Add workspace to path for mtuq
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
    
    #
    # Event parameters from USGS
    #
    event_id = 'ak2025xjbvhj'
    
    origin = Origin({
        'time': '2025-11-27T17:11:29.000000Z',
        'latitude': 61.56951904296875,
        'longitude': -150.75079345703125,
        'depth_in_m': 69433.39538574219,  # 69.43 km
    })
    
    # Earth model for Green's functions
    # Using ak135 (better for regional distances) - data resampled to 2 Hz for compatibility
    model = 'ak135'
    
    # Magnitude estimate (used for source-time function)
    magnitude = 6.0
    
    #
    # File paths
    #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(script_dir, 'waveforms', '*.[zrt]')
    path_weights = os.path.join(script_dir, 'weights.dat')
    output_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    #
    # Processing parameters for surface waves
    # (Body waves are challenging for this deep event - focusing on surface waves)
    #
    
    # For deep earthquakes, surface waves are key - use longer periods
    # ak135f_2s model supports periods >= 2s
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.02,       # 50 second period
        freq_max=0.1,        # 10 second period (well within ak135 range)
        pick_type='user_supplied',  # Use picks from weights file
        window_type='surface_wave',
        window_length=150.,  # window for surface waves
        capuaf_file=path_weights,
    )
    
    # Body waves - can use shorter periods with ak135
    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.03,       # 33 second period  
        freq_max=0.15,       # 6.7 second period
        pick_type='user_supplied',  # Use picks from weights file
        window_type='body_wave',
        window_length=30.,   # window for body waves
        capuaf_file=path_weights,
    )
    
    #
    # Misfit function configuration
    #
    
    # Surface wave misfit - allow larger time shifts for surface waves
    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-15.,
        time_shift_max=+15.,
        time_shift_groups=['ZR', 'T'],  # Group Z/R and T separately
        normalize=True,
    )
    
    # Body wave misfit
    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-5.,
        time_shift_max=+5.,
        time_shift_groups=['ZR'],
        normalize=True,
    )
    
    #
    # Station selection
    #
    print('Parsing station codes from weights file...\n')
    station_id_list = parse_station_codes(path_weights)
    print(f'Using {len(list(station_id_list))} stations\n')
    
    #
    # Moment tensor grid and source-time function
    #
    # For a full moment tensor, we search over:
    # - gamma (source type, -30 to 30 on the lune)
    # - delta (deviation from DC, -90 to 90 on the lune)
    # - kappa (strike)
    # - sigma (slip direction)
    # - h (cos(dip))
    # - M0 (seismic moment / magnitude)
    #
    
    # Grid with magnitude range around estimated magnitude
    # Start with a coarser grid for initial exploration
    grid = FullMomentTensorGridSemiregular(
        npts_per_axis=10,  # moderate resolution
        magnitudes=[5.8, 5.9, 6.0, 6.1, 6.2],
    )
    
    print(f'Grid size: {grid.size} moment tensors\n')
    
    # Source-time function (trapezoidal pulse)
    wavelet = Trapezoid(magnitude=magnitude)
    
    #
    # Read and process data
    #
    print('Reading waveform data...\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:m', 'type:velocity'])
    
    if len(data) == 0:
        print("ERROR: No data read. Check file paths and station codes.")
        sys.exit(1)
    
    print(f'Read {len(data)} stations\n')
    
    data.sort_by_distance()
    stations = data.get_stations()
    
    print('Processing data...\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)
    
    #
    # Download Green's functions from Syngine
    #
    print('Downloading Green\'s functions from Syngine...\n')
    print('(This may take several minutes for multiple stations)\n')
    
    greens = download_greens(
        stations, origin, model,
        cache_path=os.path.join(script_dir, 'greens_cache'),
    )
    
    print('Processing Green\'s functions...\n')
    greens.convolve(wavelet)
    greens_bw = greens.map(process_bw)
    greens_sw = greens.map(process_sw)
    
    #
    # Grid search
    #
    print('='*60)
    print('Starting grid search over full moment tensor space...')
    print('='*60 + '\n')
    
    print('Evaluating surface wave misfit...\n')
    results_sw = grid_search(data_sw, greens_sw, misfit_sw, origin, grid)
    
    print('Evaluating body wave misfit...\n')
    results_bw = grid_search(data_bw, greens_bw, misfit_bw, origin, grid)
    
    # Combine results (weighted sum)
    # Give more weight to surface waves for this deep event
    results = 0.3 * results_bw + 0.7 * results_sw
    
    #
    # Extract best-fitting source
    #
    print('\n' + '='*60)
    print('RESULTS')
    print('='*60 + '\n')
    
    # Index of best-fitting moment tensor
    idx = results.source_idxmin()
    
    # MomentTensor object
    best_mt = grid.get(idx)
    
    # Dictionary of lune parameters (gamma, delta, kappa, sigma, h, rho)
    lune_dict = grid.get_dict(idx)
    
    # Dictionary of Mij parameters
    mt_dict = best_mt.as_dict()
    
    # Merge all results
    merged_dict = merge_dicts(
        mt_dict, lune_dict, 
        {'M0': best_mt.moment()},
        {'Mw': best_mt.magnitude()}, 
        origin
    )
    
    # Print results
    print('Best-fitting moment tensor:')
    print(f'  Magnitude (Mw): {best_mt.magnitude():.2f}')
    print(f'  Moment (M0): {best_mt.moment():.2e} N·m')
    print()
    print('Lune/Source coordinates:')
    print(f'  v: {lune_dict["v"]:.3f}')
    print(f'  w: {lune_dict["w"]:.3f}')
    print(f'  rho (normalized moment): {lune_dict["rho"]:.3f}')
    print()
    print('Fault parameters:')
    print(f'  kappa (strike): {lune_dict["kappa"]:.1f}°')
    print(f'  sigma (slip): {lune_dict["sigma"]:.1f}°')
    print(f'  h (cos dip): {lune_dict["h"]:.3f}')
    
    # Calculate gamma and delta from v, w for interpretation
    import numpy as np
    v, w = lune_dict['v'], lune_dict['w']
    # v = gamma/-30 to 1/3, w = 3*pi/8*delta normalized
    gamma = -v * 30  # approximate conversion
    delta = w / (3*np.pi/8) * 90  # approximate conversion
    print()
    print('Interpretation:')
    print(f'  Approx. gamma (source type): {gamma:.1f}° (-30=DC, +30=CLVD)')
    print(f'  Approx. delta (DC deviation): {delta:.1f}° (0=DC, ±90=isotropic)')
    print()
    print('Moment tensor components (in N·m):')
    print(f'  Mrr: {mt_dict["Mrr"]:.2e}')
    print(f'  Mtt: {mt_dict["Mtt"]:.2e}')
    print(f'  Mpp: {mt_dict["Mpp"]:.2e}')
    print(f'  Mrt: {mt_dict["Mrt"]:.2e}')
    print(f'  Mrp: {mt_dict["Mrp"]:.2e}')
    print(f'  Mtp: {mt_dict["Mtp"]:.2e}')
    print()
    
    #
    # Generate output files
    #
    print('Generating output files...\n')
    
    # Waveform comparison plot
    try:
        plot_data_greens2(
            os.path.join(output_dir, f'{event_id}_FMT_waveforms.png'),
            data_bw, data_sw, greens_bw, greens_sw, 
            process_bw, process_sw,
            misfit_bw, misfit_sw, 
            stations, origin, best_mt, lune_dict
        )
        print(f'  Created waveform comparison plot')
    except Exception as e:
        print(f'  Warning: Could not create waveform plot: {e}')
    
    # Beachball plot
    try:
        plot_beachball(
            os.path.join(output_dir, f'{event_id}_FMT_beachball.png'),
            best_mt, stations, origin
        )
        print(f'  Created beachball plot')
    except Exception as e:
        print(f'  Warning: Could not create beachball plot: {e}')
    
    # Misfit on lune plot
    try:
        plot_misfit_lune(
            os.path.join(output_dir, f'{event_id}_FMT_misfit_lune.png'),
            results
        )
        print(f'  Created misfit lune plot')
    except Exception as e:
        print(f'  Warning: Could not create misfit lune plot: {e}')
    
    # Save JSON results
    save_json(
        os.path.join(output_dir, f'{event_id}_FMT_solution.json'),
        merged_dict
    )
    print(f'  Saved solution to {event_id}_FMT_solution.json')
    
    # Save misfit surface as NetCDF (may fail for large grids)
    try:
        results.save(os.path.join(output_dir, f'{event_id}_FMT_misfit.nc'))
        print(f'  Saved misfit surface to {event_id}_FMT_misfit.nc')
    except Exception as e:
        print(f'  Warning: Could not save misfit NetCDF: {e}')
    
    print('\n' + '='*60)
    print('INVERSION COMPLETE')
    print('='*60)
    print(f'\nResults saved to: {output_dir}')
    print()
    
    # Summary
    print('SUMMARY:')
    print(f'  Event: {event_id}')
    print(f'  Origin: {origin["time"]}')
    print(f'  Location: {origin["latitude"]:.4f}°N, {origin["longitude"]:.4f}°W')
    print(f'  Depth: {origin["depth_in_m"]/1000:.1f} km')
    print(f'  Magnitude (Mw): {best_mt.magnitude():.2f}')
    print()
