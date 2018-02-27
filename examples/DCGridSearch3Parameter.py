
import os
import sys
import numpy as np
import mtuq.io
import mtuq.greens.fk
import mtuq.misfit

from mtuq.grids import DCGridRandom
from mtuq.grid_search import grid_search_serial
from mtuq.misfit import misfit_bw, misfit_sw
from mtuq.process_data import process_data
from mtuq.util.util import Struct
from mtuq.util.wavelets import trapezoid


paths = Struct({
    'data': os.getenv('HOME')+'/'+'projects/mtuq/20090407201255351',
    'greens': os.getenv('CENTER1')+'/'+'data/wf/FK_SYNTHETICS/scak',
    })

process_bw = process_data(
    filter_type='Bandpass',
    freq_min= 0.25,
    freq_max= 0.667,
    window_length=15.,
    window_type='cap_bw',
    )

process_sw = process_data(
    filter_type='Bandpass',
    freq_min=0.025,
    freq_max=0.0625,
    window_length=150.,
    #window_type='cap_sw',
    )

misfit = {
    'body_waves': misfit_bw,
    'surface_waves': misfit_sw,
    }

process_data = {
   'body_waves': process_bw,
   'surface_waves': process_sw,
   }

grid = DCGridRandom(
    npts=20,
    Mw=4.5)



if __name__=='__main__':
    """ Carries out grid search over double-couple moment tensor parameters;
       magnitude, event depth, and event location are fixed
    """

    print 'Reading data...\n'
    data_format = 'sac'
    data = mtuq.io.read(data_format, paths.data, wildcard='*.[zrt]')
    origin = mtuq.io.get_origin(data_format, data)
    stations = mtuq.io.get_stations(data_format, data)

    print 'Processing data...\n'
    processed_data = {}
    for key in process_data:
        processed_data[key] = map(process_data[key], data, stations)

    print 'Reading Greens functions...\n'
    generator = mtuq.greens.fk.GreensTensorGenerator(paths.greens)
    greens = generator(stations, origin)
    wavelet = trapezoid(rise_time=1., delta=0.2)
    greens.convolve(wavelet)

    print 'Processing Greens functions...\n'
    processed_greens = {}
    for key in process_data:
        processed_greens[key] = greens.map(process_data[key], stations)

    print 'Carrying out grid search...\n'
    grid_search_serial(processed_data, processed_greens, misfit, grid)


