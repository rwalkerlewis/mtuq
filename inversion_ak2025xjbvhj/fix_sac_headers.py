#!/usr/bin/env python
"""
Fix SAC headers to have correct origin time
"""

import os
import glob
import obspy
from obspy import UTCDateTime

# Correct origin time
ORIGIN_TIME = UTCDateTime('2025-11-27T17:11:29.000000Z')

WAVEFORM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waveforms')

print(f"Fixing origin time in SAC headers to: {ORIGIN_TIME}")

for filepath in glob.glob(os.path.join(WAVEFORM_DIR, '*.[zrt]')):
    st = obspy.read(filepath, format='sac')
    for tr in st:
        # Set correct origin time
        tr.stats.sac['nzyear'] = ORIGIN_TIME.year
        tr.stats.sac['nzjday'] = ORIGIN_TIME.julday
        tr.stats.sac['nzhour'] = ORIGIN_TIME.hour
        tr.stats.sac['nzmin'] = ORIGIN_TIME.minute
        tr.stats.sac['nzsec'] = ORIGIN_TIME.second
        tr.stats.sac['nzmsec'] = int(ORIGIN_TIME.microsecond / 1000)
    
    st.write(filepath, format='SAC')

print(f"Fixed {len(glob.glob(os.path.join(WAVEFORM_DIR, '*.[zrt]')))} files")
