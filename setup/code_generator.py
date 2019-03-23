

Imports="""
import os
import sys
import numpy as np

from os.path import join
from mtuq import read, get_greens_tensors, open_db
from mtuq.grid import DoubleCoupleGridRandom
from mtuq.grid_search.mpi import grid_search_mpi
from mtuq.cap.misfit import Misfit
from mtuq.cap.process_data import ProcessData
from mtuq.cap.util import Trapezoid
from mtuq.util.plot import plot_beachball, plot_data_greens_mt
from mtuq.util.util import path_mtuq


"""


Docstring_SerialGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 20 minutes. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #

"""


Docstring_GridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple.py
    #
    # For a simpler example, see SerialGridSearch.DoubleCouple.py, 
    # which runs the same inversion in serial
    #

"""


Docstring_CapStyleGridSearch_DoubleCouple="""
if __name__=='__main__':
    #
    # CAP-style Double-couple inversion example
    # 
    # Carries out grid search over 50,000 randomly chosen double-couple 
    # moment tensors
    #
    # USAGE
    #   mpirun -n <NPROC> python CapStyleGridSearch.DoubleCouple.py
    #

"""


Docstring_GridSearch_DoubleCoupleMagnitudeDepth="""
if __name__=='__main__':
    #
    # Double-couple inversion example
    #   
    # Carries out grid search over source orientation, magnitude, and depth
    #   
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.DoubleCouple+Magnitude+Depth.py
    #   

"""


Docstring_GridSearch_FullMomentTensor="""
if __name__=='__main__':
    #
    # Full moment tensor inversion example
    #   
    # Carries out grid search over all moment tensor parameters except
    # magnitude 
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

"""


Docstring_BenchmarkCAP="""
if __name__=='__main__':
    #
    # Given seven "fundamental" moment tensors, generates MTUQ synthetics and
    # compares with corresponding CAP/FK synthetics
    #
    # Before running this script, it is necessary to unpack the CAP/FK 
    # synthetics using data/tests/unpack.bash
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple3.py,
    # except here we consider only seven grid points rather than an entire
    # grid, and here the final plots are a comparison of MTUQ and CAP/FK 
    # synthetics rather than a comparison of data and synthetics
    #
    # Because of the idiosyncratic way CAP implements source-time function
    # convolution, it's not expected that CAP and MTUQ synthetics will match 
    # exactly. CAP's "conv" function results in systematic magnitude-
    # dependent shifts between origin times and arrival times. We deal with 
    # this by applying magnitude-dependent time-shifts to MTUQ synthetics 
    # (which normally lack such shifts) at the end of the benchmark. Even with
    # this correction, the match will not be exact because CAP applies the 
    # shifts before tapering and MTUQ after tapering. The resulting mismatch 
    # will usually be apparent in body-wave windows, but not in surface-wave 
    # windows
    #
    # Note that CAP works with dyne,cm and MTUQ works with N,m, so to make
    # comparisons we convert CAP output from the former to the latter
    #
    # The CAP/FK synthetics used in the comparison were generated by 
    # uafseismo/capuaf:46dd46bdc06e1336c3c4ccf4f99368fe99019c88
    # using the following commands
    #
    # source #0 (explosion):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/1.178/90/45/90 20090407201255351
    #
    # source #1 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/90/45/90 20090407201255351
    #
    # source #2 (on-diagonal)
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/180 20090407201255351
    #
    # source #3 (on-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R-0.333/0.972/45/90/0 20090407201255351
    #
    # source #4 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/90/90 20090407201255351
    #
    # source #5 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/90/0/0 20090407201255351
    #
    # source #6 (off-diagonal):
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1 -R0/0/0/90/180 20090407201255351
    #

"""


Docstring_IntegrationTest="""
if __name__=='__main__':
    #
    #
    # This script is similar to examples/SerialGridSearch.DoubleCouple3.py,
    # except here we use a coarser grid, and at the end we assert that the test
    # result equals the expected result
    #
    # The compare against CAP/FK
    # cap.pl -H0.02 -P1/15/60 -p1 -S2/10/0 -T15/150 -D1/1/0.5 -C0.1/0.333/0.025/0.0625 -Y1 -Zweight_test.dat -Mscak_34 -m4.5 -I1/1/10/10/10 -R0/0/0/0/0/360/0/90/-180/180 20090407201255351

"""


Argparse_BenchmarkCAP="""
    from mtuq.cap.util import\\
        get_synthetics_cap, get_synthetics_mtuq,\\
        get_data_cap, compare_cap_mtuq


    # by default, the script runs with figure generation and error checking
    # turned on; these can be turned off through argparse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)


    # the following directories correspond to the moment tensors in the list 
    # "grid" below
    paths = []
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/0')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/1')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/2')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/3')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/4')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/5')]
    paths += [join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/20090407201255351/6')]

"""


PathsComments="""
    #
    # Here we specify the data used for the inversion. The event is an 
    # Mw~4 Alaska earthquake
    #
"""


Paths_Syngine="""
    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'ak135'

"""


Paths_FK="""
    path_greens=  join(path_mtuq(), 'data/tests/benchmark_cap_mtuq/greens/scak')
    path_data=    join(path_mtuq(), 'data/examples/20090407201255351/*.[zrt]')
    path_weights= join(path_mtuq(), 'data/examples/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'

"""



DataProcessingComments="""
    #
    # Body- and surface-wave data are processed separately and held separately 
    # in memory
    #
"""


DataProcessingDefinitions="""
    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_taup_model',
        taup_model=model,
        window_type='cap_bw',
        window_length=15.,
        padding_length=2.,
        weight_type='cap_bw',
        cap_weight_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_taup_model',
        taup_model=model,
        window_type='cap_sw',
        window_length=150.,
        padding_length=10.,
        weight_type='cap_sw',
        cap_weight_file=path_weights,
        )

    process_data = {
       'body_waves': process_bw,
       'surface_waves': process_sw,
       }

"""


MisfitComments="""
    #
    # We define misfit as a sum of indepedent body- and surface-wave 
    # contributions
    #
"""


MisfitDefinitions="""
    misfit_bw = Misfit(
        time_shift_max=2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        time_shift_max=10.,
        time_shift_groups=['ZR','T'],
        )

    misfit = {
        'body_waves': misfit_bw,
        'surface_waves': misfit_sw,
        }

"""


Grid_DoubleCouple="""
    #
    # Next we specify the source parameter grid
    #

    grid = DoubleCoupleGridRandom(
        npts=50000,
        moment_magnitude=4.5)

    wavelet = Trapezoid(
        moment_magnitude=4.5)

"""


Grid_DoubleCoupleMagnitudeDepth="""
    #
    # Next we specify the source parameter grid
    #

    grid = DoubleCoupleGridRandom(
        npts=50000,
        moment_magnitude=4.5)

    origins = OriginGrid(depth=np.arange(2500.,20000.,2500.),
        latitude=origin.latitude,
        longitude=origin.longitude)

    wavelet = Trapezoid(
        moment_magnitude=4.5)

"""


Grid_FullMomentTensor="""
    #
    # Next we specify the source parameter grid
    #

    grid = FullMomentTensorGridRandom(
        npts=1000000,
        moment_magnitude=4.5)

    wavelet = Trapezoid(
        moment_magnitude=4.5)

"""


Grid_BenchmarkCAP="""
    #
    # Next we specify the source parameter grid
    #

    grid = [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (on-diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (on-diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (on-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 1., 0., 0.]), # source 4 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 1., 0.]), # source 5 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 0., 1.]), # source 6 (off-diagonal)
       ]

    Mw = 4.5
    M0 = 10.**(1.5*Mw + 9.1) # units: N-m
    for mt in grid:
        mt *= np.sqrt(2)*M0

    wavelet = Trapezoid(
        moment_magnitude=Mw)

"""


Grid_IntegrationTest="""
    grid = DoubleCoupleGridRegular(
        moment_magnitude=4.5, 
        npts_per_axis=10)

    wavelet = Trapezoid(
        moment_magnitude=4.5)


"""




Main_SerialGridSearch="""
    #
    # The main I/O work starts now
    #

    print 'Reading data...\\n'
    data = read(path_data, format='sac',
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origin()


    print 'Processing data...\\n'
    processed_data = {}
    for key in ['body_waves', 'surface_waves']:
        processed_data[key] = data.map(process_data[key])
    data = processed_data


    print 'Downloading Greens functions...\\n'
    greens = get_greens_tensors(stations, origin, model=model)



    print 'Processing Greens functions...\\n'
    greens.convolve(wavelet)
    processed_greens = {}
    for key in ['body_waves', 'surface_waves']:
        processed_greens[key] = greens.map(process_data[key])
    greens = processed_greens


    #
    # The main computational work starts nows
    #

    print 'Carrying out grid search...\\n'
    results = grid_search_serial(data, greens, misfit, grid)


    print 'Saving results...\\n'
    #grid.save(event_name+'.h5', {'misfit': results})
    best_mt = grid.get(results.argmin())


    print 'Plotting waveforms...\\n'
    plot_data_greens_mt(event_name+'.png', data, greens, best_mt, misfit)
    plot_beachball(event_name+'_beachball.png', best_mt)


"""



Main_GridSearch_DoubleCouple="""
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print 'Reading data...\\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origin = data.get_origin()

        print 'Processing data...\\n'
        processed_data = {}
        for key in ['body_waves', 'surface_waves']:
            processed_data[key] = data.map(process_data[key])
        data = processed_data

        print 'Reading Greens functions...\\n'
        greens = get_greens_tensors(stations, origin, model=model)

        print 'Processing Greens functions...\\n'
        greens.convolve(wavelet)
        processed_greens = {}
        for key in ['body_waves', 'surface_waves']:
            processed_greens[key] = greens.map(process_data[key])
        greens = processed_greens

    else:
        data = None
        greens = None

    data = comm.bcast(data, root=0)
    greens = comm.bcast(greens, root=0)


    #
    # The main computational work starts now
    #

    if comm.rank==0:
        print 'Carrying out grid search...\\n'
    results = grid_search_mpi(data, greens, misfit, grid)
    results = comm.gather(results, root=0)


    if comm.rank==0:
        print 'Saving results...\\n'
        results = np.concatenate(results)
        #grid.save(event_name+'.h5', {'misfit': results})
        best_mt = grid.get(results.argmin())


    if comm.rank==0:
        print 'Plotting waveforms...\\n'
        plot_data_greens_mt(event_name+'.png', data, greens, best_mt, misfit)
        plot_beachball(event_name+'_beachball.png', best_mt)


"""



Main_GridSearch_DoubleCoupleMagnitudeDepth="""
    #
    # The main work of the grid search starts now
    #
    from mpi4py import MPI
    comm = MPI.COMM_WORLD


    if comm.rank==0:
        print 'Reading data...\\n'
        data = read(path_data, format='sac', 
            event_id=event_name,
            tags=['units:cm', 'type:velocity']) 

        data.sort_by_distance()

        stations = data.get_stations()
        origin = data.get_origin()

        print 'Processing data...\\n'
        processed_data = {}
        for key in ['body_waves', 'surface_waves']:
            processed_data[key] = data.map(process_data[key])
        data = processed_data
    else:
        data = None

    data = comm.bcast(data, root=0)


   for origin, magnitude in cross(origins, magnitudes):
        if comm.rank==0:
            print 'Downloading Greens functions...\\n'
            greens = get_greens_tensors(stations, origin, model=model)

            print 'Processing Greens functions...\\n'
            wavelet = Trapezoid(magnitude)
            greens.convolve(wavelet)

            processed_greens = {}
            for key in ['body_waves', 'surface_waves']:
                processed_greens[key] = greens.map(process_data[key])
            greens = processed_greens

        else:
            greens = None

        greens = comm.bcast(greens, root=0)


        if comm.rank==0:
            print 'Carrying out grid search...\\n'
        results = grid_search_mpi(data, greens, misfit, grid)
        results = comm.gather(results, root=0)


        if comm.rank==0:
            print 'Saving results...\\n'
            results = np.concatenate(results)
            #grid.save(event_name+'.h5', {'misfit': results})


        if comm.rank==0:
            print 'Plotting waveforms...\\n'
            plot_data_greens_mt(event_name+'.png', data, greens, best_mt, misfit)
            plot_beachball(event_name+'_beachball.png', best_mt)



"""


Main_BenchmarkCAP="""
    #
    # The benchmark starts now
    #

    print 'Reading data...\\n'
    data = read(path_data, format='sac', 
        event_id=event_name,
        tags=['units:cm', 'type:velocity']) 

    data.sort_by_distance()

    stations = data.get_stations()
    origin = data.get_origin()


    print 'Processing data...\\n'
    processed_data = {}
    for key in ['body_waves', 'surface_waves']:
        processed_data[key] = data.map(process_data[key])
    data = processed_data


    print 'Reading Greens functions...\\n'
    db = open_db(path_greens, format='FK')
    greens = db.get_greens_tensors(stations, origin)

    print 'Processing Greens functions...\\n'
    greens.convolve(wavelet)
    processed_greens = {}
    for key in ['body_waves', 'surface_waves']:
        processed_greens[key] = greens.map(process_data[key])
    greens = processed_greens

    print 'Comparing waveforms...'

    for _i, mt in enumerate(grid):
        print ' %d of %d' % (_i+1, len(grid))

        name = model+'_34_'+event_name
        synthetics_cap = get_synthetics_cap(data, paths[_i], name)
        synthetics_mtuq = get_synthetics_mtuq(data, greens, mt)

        if run_figures:
            filename = 'cap_vs_mtuq_'+str(_i)+'.png'
            plot_data_synthetics(filename, synthetics_cap, synthetics_mtuq)

        if run_checks:
            compare_cap_mtuq(synthetics_cap, synthetics_mtuq)

    if run_figures:
        # "bonus" figure comparing how CAP processes observed data with how
        # MTUQ processes observed data
        data_mtuq = data
        data_cap = get_data_cap(data, paths[0], name)
        filename = 'cap_vs_mtuq_data.png'
        plot_data_synthetics(filename, data_cap, data_mtuq, normalize=False)

"""



if __name__=='__main__':
    import os
    import re

    from mtuq.util.util import path_mtuq, replace
    os.chdir(path_mtuq())


    with open('examples/GridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_GridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(Main_GridSearch_DoubleCouple)


    with open('examples/GridSearch.DoubleCouple+Magnitude+Depth.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_GridSearch_DoubleCoupleMagnitudeDepth)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCoupleMagnitudeDepth)
        file.write(Main_GridSearch_DoubleCoupleMagnitudeDepth)


    with open('examples/GridSearch.FullMomentTensor.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(Imports)
        file.write(Docstring_GridSearch_FullMomentTensor)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_FullMomentTensor)
        file.write(Main_GridSearch_DoubleCouple)


    with open('examples/SerialGridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'grid_search_mpi',
            'grid_search_serial',
            ))
        file.write(Docstring_SerialGridSearch_DoubleCouple)
        file.write(PathsComments)
        file.write(Paths_Syngine)
        file.write(DataProcessingComments)
        file.write(DataProcessingDefinitions)
        file.write(MisfitComments)
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(Main_SerialGridSearch)


    with open('tests/test_grid_search.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'grid_search_mpi',
            'grid_search_serial',
            'DoubleCoupleGridRandom',
            'DoubleCoupleGridRegular',
            ))
        file.write(Docstring_IntegrationTest)
        file.write(Paths_Syngine)
        file.write(DataProcessingDefinitions)
        file.write(MisfitDefinitions)
        file.write(Grid_IntegrationTest)
        file.write(Main_SerialGridSearch)


    with open('tests/benchmark_cap.py', 'w') as file:
        file.write(
            replace(
            Imports,
            'grid_search_mpi',
            'grid_search_serial',
            'syngine',
            'fk',
            'plot_data_greens_mt',
            'plot_data_synthetics',
            ))
        file.write(Docstring_BenchmarkCAP)
        file.write(Argparse_BenchmarkCAP)
        file.write(Paths_FK)
        file.write(
            replace(
            DataProcessingDefinitions,
            'padding_length=.*',
            'padding_length=0,',
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(
            replace(
            MisfitDefinitions,
            'time_shift_max=.*',
            'time_shift_max=0.,',
            ))
        file.write(Grid_BenchmarkCAP)
        file.write(Main_BenchmarkCAP)


    with open('setup/chinook/examples/CapStyleGridSearch.DoubleCouple.py', 'w') as file:
        file.write("#!/usr/bin/env python\n")
        file.write(
            replace(
            Imports,
            'syngine',
            'fk'
            ))
        file.write(Docstring_CapStyleGridSearch_DoubleCouple)
        file.write(
            replace(
            Paths_FK,
           r"path_greens=.*",
           r"path_greens= '/import/c1/ERTHQUAK/rmodrak/wf/FK_synthetics/scak'",
            ))
        file.write(
            replace(
            DataProcessingDefinitions,
            'pick_type=.*',
            "pick_type='from_fk_metadata',",
            'taup_model=.*,',
            'fk_database=path_greens,',
            ))
        file.write(MisfitDefinitions)
        file.write(Grid_DoubleCouple)
        file.write(
            replace(
            Main_GridSearch_DoubleCouple,
            'print ''Downloading Greens functions...\\n''',
            'print ''Reading Greens functions...\\n''',
            'greens = get_greens_tensors\(stations, origin, model=model\)',
            'db = open_db(path_greens, format=\'FK\', model=model)\n        '
           +'greens = db.get_greens_tensors(stations, origin)',
            ))

