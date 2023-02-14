#! /usr/bin/env python

# Obspy processing of 2019321 Moroccan event data
# Use with pysep environment

import numpy as np
import os
from obspy import read, read_inventory
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from pysep import Pysep
from pysep.utils.cap_sac import (append_sac_headers, write_cap_weights_files,
                                 format_sac_header_w_taup_traveltimes,
                                 format_sac_headers_post_rotation)


# Day info
t1 = UTCDateTime('2019-11-17T00:00:00.000000Z')
t2 = UTCDateTime('2019-11-17T23:59:59.950000Z')
fdsn_client = Client('GEOFON')

# Add event - Mainshock ()
main_time = UTCDateTime('2019-11-17T08:39:09')
m_t1 = main_time - 30
m_t2 = main_time + 30

cat = fdsn_client.get_events(starttime=m_t1, endtime=m_t2)
event = cat[0]

# Load waveforms
st_full = read('2019321_ZNE/*')

# Cut waveforms to time of interest

st = st_full.copy().slice(main_time-20, main_time + 600)

# Add network codes
for tr in st:
    # Most are network MO
    tr.stats.network = 'MO'

# Not sure which network MD is
for tr in st.select(station='MD31'):
    tr.stats.network = 'MD'

# WM Network
WM_STA = ['AVE','TIO']

for sta in WM_STA:
    for tr in st.select(station=sta):
        tr.stats.network = 'WM'

# Remove stations without response files

no_resp = ['RTC','TFR','TIS']

for sta in no_resp:
    for tr in st.select(station=sta):
        st.remove(tr)

# Downsample to 20 Hz
for tr in st:
    if tr.stats.sampling_rate == 40.0:
        tr.decimate(factor=2)
        tr.stats.sac.delta = np.round(tr.stats.sac.delta * 2, 5)


# Get instrument response
inv_MO = read_inventory('RESP/RESP.*')
inv_WM_AVE = fdsn_client.get_stations(network = 'WM', 
                               station = 'AVE',
                               channel = 'BH*',
                               starttime = t1,
                               endtime = t2,
                               level = 'response')

inv_WM_TIO = fdsn_client.get_stations(network = 'WM', 
                               station = 'TIO',
                               channel = 'BH*',
                               starttime = t1,
                               endtime = t2,
                               level = 'response')

inv = inv_MO + inv_WM_AVE + inv_WM_TIO

# Remove instrument response
st.remove_response(inventory=inv)

# Load into PySEP. MTUQ requires ZRT orientation for seismograms

# Rotate to RTZ orientation
for tr in st:
    tr.stats.back_azimuth = tr.stats.sac.baz
st.rotate('NE->RT')
st = format_sac_headers_post_rotation(st)

# Write data    
event_tag = '2019321'
for tr in st:
    tr.stats.sac.kevnm = event_tag

for tr in st:
    # if self._legacy_naming:
    # Legacy: e.g., 20000101000000.NN.SSS.LL.CC.c
    _trace_id = f"{tr.get_id()[:-1]}.{tr.get_id()[-1].lower()}"
    tag = f"{event_tag}.{_trace_id}"
    # else:
    #     # New style: e.g., 2000-01-01T000000.NN.SSS.LL.CCC.sac
    #     tag = f"{self.event_tag}.{tr.get_id()}.sac"
    fid = os.path.join('2019321_RTZ/', tag)
    # logger.debug(os.path.basename(fid))
    tr.write(fid, format="SAC")

# Write config file
write_cap_weights_files(st, path_out="./2019321_RTZ")