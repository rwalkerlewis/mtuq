#!/usr/bin/env python
"""
Create CAP-format weights file for MTUQ inversion
Select best stations based on distance and azimuthal coverage
"""

import os
import glob
import obspy
import numpy as np
from obspy.geodetics import gps2dist_azimuth

# Event parameters
EVENT_ID = 'ak2025xjbvhj'
ORIGIN_LAT = 61.56951904296875
ORIGIN_LON = -150.75079345703125

# Station selection criteria
MIN_DIST_KM = 50.0     # Minimum distance
MAX_DIST_KM = 600.0    # Maximum distance for surface waves
MAX_STATIONS = 50      # Maximum number of stations to use

# Input/output paths
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
WAVEFORM_DIR = os.path.join(OUTPUT_DIR, 'waveforms')
WEIGHTS_FILE = os.path.join(OUTPUT_DIR, 'weights.dat')


def get_station_info():
    """Get station information from SAC files"""
    stations = {}
    
    for filepath in glob.glob(os.path.join(WAVEFORM_DIR, '*.z')):
        try:
            st = obspy.read(filepath, format='sac')
            tr = st[0]
            
            # Get station info from filename and SAC headers
            filename = os.path.basename(filepath)
            parts = filename.replace('.z', '').split('.')
            
            if len(parts) >= 3:
                net = parts[0]
                sta = parts[1]
                loc = parts[2] if len(parts) > 2 else ''
                
                station_id = f"{net}.{sta}.{loc}"
                
                # Get station coordinates from SAC headers
                sta_lat = tr.stats.sac.get('stla', None)
                sta_lon = tr.stats.sac.get('stlo', None)
                
                if sta_lat is None or sta_lon is None:
                    continue
                
                # Calculate distance and azimuth
                dist_m, az, baz = gps2dist_azimuth(ORIGIN_LAT, ORIGIN_LON, sta_lat, sta_lon)
                dist_km = dist_m / 1000.0
                
                stations[station_id] = {
                    'net': net,
                    'sta': sta,
                    'loc': loc,
                    'lat': sta_lat,
                    'lon': sta_lon,
                    'dist_km': dist_km,
                    'az': az,
                    'baz': baz
                }
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return stations


def select_stations(stations):
    """Select best stations based on distance and azimuthal coverage"""
    
    # Filter by distance
    filtered = {k: v for k, v in stations.items() 
                if MIN_DIST_KM <= v['dist_km'] <= MAX_DIST_KM}
    
    if len(filtered) <= MAX_STATIONS:
        return filtered
    
    # Group by azimuth bins (36 bins of 10 degrees each)
    n_bins = 36
    bin_size = 360.0 / n_bins
    
    az_bins = {i: [] for i in range(n_bins)}
    for station_id, info in filtered.items():
        bin_idx = int(info['az'] / bin_size) % n_bins
        az_bins[bin_idx].append((station_id, info))
    
    # Sort each bin by distance and select closest
    selected = {}
    
    # First pass: select one station per azimuth bin
    for bin_idx in range(n_bins):
        if az_bins[bin_idx]:
            # Sort by distance
            az_bins[bin_idx].sort(key=lambda x: x[1]['dist_km'])
            # Select closest
            station_id, info = az_bins[bin_idx][0]
            selected[station_id] = info
    
    # Second pass: fill remaining slots with next closest stations
    all_remaining = []
    for bin_idx in range(n_bins):
        for station_id, info in az_bins[bin_idx][1:]:
            if station_id not in selected:
                all_remaining.append((station_id, info))
    
    all_remaining.sort(key=lambda x: x[1]['dist_km'])
    
    for station_id, info in all_remaining:
        if len(selected) >= MAX_STATIONS:
            break
        selected[station_id] = info
    
    return selected


def write_weights_file(stations):
    """Write CAP-format weights file"""
    
    # Sort by distance
    sorted_stations = sorted(stations.items(), key=lambda x: x[1]['dist_km'])
    
    with open(WEIGHTS_FILE, 'w') as f:
        f.write("#  event_id.net.sta.loc.ch          offset_km    weights               P_pick  bw_len    S_pick  sw_len    rw_static  lw_static\n")
        
        for station_id, info in sorted_stations:
            net = info['net']
            sta = info['sta']
            loc = info['loc']
            dist_km = info['dist_km']
            
            # Build station code string
            code = f"{EVENT_ID}.{net}.{sta}.{loc}.BH"
            
            # Estimate P and S picks based on distance
            # Approximate velocities for regional distances
            p_vel = 8.0  # km/s
            s_vel = 4.5  # km/s
            
            p_pick = dist_km / p_vel
            s_pick = dist_km / s_vel
            
            # Body wave length and surface wave length
            bw_len = 0.0  # Use default
            sw_len = 0.0  # Use default
            
            # Weights: body_wave_Z, body_wave_R, surface_wave_Z, surface_wave_R, surface_wave_T
            # For regional earthquakes, use both body and surface waves
            # Disable body waves for distant stations
            if dist_km < 300:
                bw_z = 1
                bw_r = 1
            else:
                bw_z = 0
                bw_r = 0
            
            # Enable surface waves for all
            sw_z = 1
            sw_r = 1
            sw_t = 1
            
            # Write line
            f.write(f"   {code:40s} {int(dist_km):3d}           {bw_z}   {bw_r}   {sw_z}   {sw_r}   {sw_t}     {p_pick:.2f}    {bw_len:.1f}        {s_pick:.2f}    {sw_len:.1f}        0.         0.\n")
    
    print(f"Created weights file: {WEIGHTS_FILE}")
    print(f"Selected {len(sorted_stations)} stations")


if __name__ == '__main__':
    print("Getting station information...")
    stations = get_station_info()
    print(f"Found {len(stations)} stations with waveforms")
    
    print("\nSelecting best stations...")
    selected = select_stations(stations)
    print(f"Selected {len(selected)} stations")
    
    print("\nWriting weights file...")
    write_weights_file(selected)
    
    print("\nStation distances:")
    for station_id, info in sorted(selected.items(), key=lambda x: x[1]['dist_km']):
        print(f"  {station_id}: {info['dist_km']:.1f} km, az={info['az']:.1f}°")
