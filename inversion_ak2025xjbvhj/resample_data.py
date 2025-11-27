#!/usr/bin/env python
"""
Resample waveform data to higher sampling rate for Syngine compatibility
ak135f_2s model requires dt < 1.0s (min_period = 2s)
"""

import os
import glob
import obspy
from scipy import signal

WAVEFORM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waveforms')
NEW_SAMPLING_RATE = 2.0  # 2 Hz (dt = 0.5s)

print(f"Resampling data to {NEW_SAMPLING_RATE} Hz (dt = {1/NEW_SAMPLING_RATE}s)")

count = 0
for filepath in glob.glob(os.path.join(WAVEFORM_DIR, '*.[zrt]')):
    st = obspy.read(filepath, format='sac')
    tr = st[0]
    
    old_rate = tr.stats.sampling_rate
    if old_rate != NEW_SAMPLING_RATE:
        # Upsample using interpolation
        tr.interpolate(sampling_rate=NEW_SAMPLING_RATE, method='lanczos', a=20)
        st.write(filepath, format='SAC')
        count += 1

print(f"Resampled {count} files to {NEW_SAMPLING_RATE} Hz")

# Verify
st = obspy.read(glob.glob(os.path.join(WAVEFORM_DIR, '*.z'))[0], format='sac')
print(f"New sampling rate: {st[0].stats.sampling_rate} Hz")
print(f"New dt: {st[0].stats.delta}s")
