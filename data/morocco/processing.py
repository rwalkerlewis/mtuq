#! /usr/bin/env python

# Obspy processing of 2019321 Moroccan event data

from obspy import read, read_inventory
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

st = read('2019321/*')
t1 = UTCDateTime('2019-11-17T00:00:00.000000Z')
t2 = UTCDateTime('2019-11-17T23:59:59.950000Z')
fdsn_client = Client('IRIS')

# Add network codes
for tr in st:
    if tr.stats.station != 'MD31':
        tr.stats.network = 'MO'
    elif tr.stats.station == 'MD31':
        tr.stats.network = 'MD'

# Get instrument response


# Remove instrument response
inv = read_inventory('RESP/RESP.*')
# inv = fdsn_client.get_stations(network = 'MO', 
#                                station = '*',
#                                channel = 'BH*',
#                                starttime = t1,
#                                endtime = t2,
#                                level = 'response')