#!/usr/bin/env python
"""
Download seismic waveform data for earthquake ak2025xjbvhj
M6.0 earthquake near Susitna, Alaska on 2025-11-27
"""

import os
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth

# Event parameters from USGS
EVENT_ID = 'ak2025xjbvhj'
ORIGIN_TIME = UTCDateTime('2025-11-27T17:11:29.000Z')
ORIGIN_LAT = 61.56951904296875
ORIGIN_LON = -150.75079345703125
ORIGIN_DEPTH_KM = 69.43339538574219  # km
MAGNITUDE = 6.0

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
WAVEFORM_DIR = os.path.join(OUTPUT_DIR, 'waveforms')
os.makedirs(WAVEFORM_DIR, exist_ok=True)

# Data parameters
STARTTIME = ORIGIN_TIME - 60  # 60 seconds before origin
ENDTIME = ORIGIN_TIME + 600   # 600 seconds after origin
MIN_RADIUS_DEG = 0.5          # minimum distance in degrees
MAX_RADIUS_DEG = 10.0         # maximum distance in degrees (~1000 km)

# Initialize FDSN clients
print("Initializing FDSN clients...")
iris_client = Client("IRIS")

def get_stations():
    """Get available stations within radius of event"""
    print(f"Searching for stations within {MAX_RADIUS_DEG} degrees...")
    
    inventory = iris_client.get_stations(
        starttime=ORIGIN_TIME,
        endtime=ORIGIN_TIME,
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LON,
        minradius=MIN_RADIUS_DEG,
        maxradius=MAX_RADIUS_DEG,
        channel="BH?,HH?",  # Broadband channels
        level="channel"
    )
    
    return inventory


def download_waveforms(inventory):
    """Download waveforms for all stations in inventory"""
    print("Downloading waveforms...")
    
    all_streams = Stream()
    stations_downloaded = 0
    
    for network in inventory:
        for station in network:
            net_code = network.code
            sta_code = station.code
            sta_lat = station.latitude
            sta_lon = station.longitude
            
            # Calculate distance and azimuth
            dist_m, az, baz = gps2dist_azimuth(ORIGIN_LAT, ORIGIN_LON, sta_lat, sta_lon)
            dist_km = dist_m / 1000.0
            
            try:
                # Download waveforms
                st = iris_client.get_waveforms(
                    network=net_code,
                    station=sta_code,
                    location="*",
                    channel="BH?,HH?",
                    starttime=STARTTIME,
                    endtime=ENDTIME,
                    attach_response=True
                )
                
                if len(st) > 0:
                    # Add event/station metadata
                    for tr in st:
                        tr.stats.sac = {}
                        tr.stats.sac['evla'] = ORIGIN_LAT
                        tr.stats.sac['evlo'] = ORIGIN_LON
                        tr.stats.sac['evdp'] = ORIGIN_DEPTH_KM
                        tr.stats.sac['stla'] = sta_lat
                        tr.stats.sac['stlo'] = sta_lon
                        tr.stats.sac['nzyear'] = ORIGIN_TIME.year
                        tr.stats.sac['nzjday'] = ORIGIN_TIME.julday
                        tr.stats.sac['nzhour'] = ORIGIN_TIME.hour
                        tr.stats.sac['nzmin'] = ORIGIN_TIME.minute
                        tr.stats.sac['nzsec'] = ORIGIN_TIME.second
                        tr.stats.sac['dist'] = dist_km
                        tr.stats.sac['az'] = az
                        tr.stats.sac['baz'] = baz
                    
                    all_streams += st
                    stations_downloaded += 1
                    print(f"  Downloaded {net_code}.{sta_code} ({dist_km:.1f} km, {len(st)} traces)")
                    
            except Exception as e:
                print(f"  Failed for {net_code}.{sta_code}: {str(e)[:50]}")
    
    print(f"\nDownloaded {stations_downloaded} stations, {len(all_streams)} traces total")
    return all_streams


def process_waveforms(st):
    """Process waveforms: detrend, remove response, rotate to ZRT"""
    print("\nProcessing waveforms...")
    
    # Remove response to velocity
    st_processed = st.copy()
    
    # Group traces by station
    stations = set()
    for tr in st_processed:
        stations.add(f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}")
    
    processed_streams = []
    
    for station_id in sorted(stations):
        net, sta, loc = station_id.split('.')
        st_sta = st_processed.select(network=net, station=sta, location=loc)
        
        if len(st_sta) < 3:
            print(f"  Skipping {station_id} - incomplete components")
            continue
            
        try:
            # Detrend and taper
            st_sta.detrend('demean')
            st_sta.detrend('linear')
            st_sta.taper(max_percentage=0.05)
            
            # Remove instrument response to velocity
            st_sta.remove_response(output='VEL', pre_filt=[0.005, 0.01, 40, 45])
            
            # Resample to 1 Hz for surface waves (save bandwidth)
            st_sta.resample(1.0)
            
            # Rotate to ZNE if needed, then to ZRT
            try:
                # Try to rotate NEZ to ZNE first
                st_sta._rotate_to_zne(inventory=None, components=['Z', 'N', 'E'])
            except:
                pass
            
            # Get back azimuth for rotation
            baz = st_sta[0].stats.sac.get('baz', None)
            if baz is not None:
                try:
                    st_sta.rotate(method='NE->RT', back_azimuth=baz)
                except Exception as e:
                    print(f"  Could not rotate {station_id}: {e}")
                    continue
            
            processed_streams.append(st_sta)
            print(f"  Processed {station_id}")
            
        except Exception as e:
            print(f"  Error processing {station_id}: {e}")
    
    return processed_streams


def save_waveforms(processed_streams):
    """Save processed waveforms as SAC files"""
    print("\nSaving waveforms...")
    
    for st in processed_streams:
        for tr in st:
            net = tr.stats.network
            sta = tr.stats.station
            loc = tr.stats.location or ''
            cha = tr.stats.channel
            
            # Get component letter (Z, R, T)
            comp = cha[-1].lower()
            
            # Create filename
            filename = f"{net}.{sta}.{loc}.{comp}"
            filepath = os.path.join(WAVEFORM_DIR, filename)
            
            # Write SAC file
            tr.write(filepath, format='SAC')
    
    print(f"Saved waveforms to {WAVEFORM_DIR}")


def create_weights_file(processed_streams):
    """Create weights file for MTUQ"""
    print("\nCreating weights file...")
    
    weights_file = os.path.join(OUTPUT_DIR, 'weights.dat')
    
    with open(weights_file, 'w') as f:
        f.write("# Station weights file for MTUQ\n")
        f.write("# net.sta.loc  dist(km)  weights(BHZ BHR BHT surface BHZ BHR BHT)\n")
        
        for st in processed_streams:
            if len(st) == 0:
                continue
                
            tr = st[0]
            net = tr.stats.network
            sta = tr.stats.station
            loc = tr.stats.location or ''
            dist_km = tr.stats.sac.get('dist', 0)
            
            station_id = f"{net}.{sta}.{loc}"
            
            # Default weights: all components enabled
            # Format: bodywave_Z, bodywave_R, bodywave_T, surface_Z, surface_R, surface_T
            f.write(f"{station_id}  {dist_km:.1f}  1 1 0 1 1 1\n")
    
    print(f"Created weights file: {weights_file}")


if __name__ == '__main__':
    print("="*60)
    print(f"Downloading data for earthquake {EVENT_ID}")
    print(f"M{MAGNITUDE} - {ORIGIN_TIME}")
    print(f"Location: {ORIGIN_LAT:.4f}N, {ORIGIN_LON:.4f}W, {ORIGIN_DEPTH_KM:.1f} km")
    print("="*60)
    
    # Get station inventory
    inventory = get_stations()
    print(f"Found {sum(len(net) for net in inventory)} stations")
    
    # Download waveforms
    streams = download_waveforms(inventory)
    
    if len(streams) == 0:
        print("No waveforms downloaded. Exiting.")
        exit(1)
    
    # Process waveforms
    processed = process_waveforms(streams)
    
    if len(processed) == 0:
        print("No waveforms successfully processed. Exiting.")
        exit(1)
    
    # Save waveforms
    save_waveforms(processed)
    
    # Create weights file
    create_weights_file(processed)
    
    print("\nData download complete!")
