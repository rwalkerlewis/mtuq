
import numpy as np

import mtuq
from mtuq.util.math import radiation_coef
from mtuq.util import Null, iterable, warn
from mtuq.misfit.waveform.level2 import _to_array, _type
from obspy.taup import TauPyModel
from mtuq.util.signal import m_to_deg


class PolarityMisfit(object):
    """ Polarity misfit function
    """

    def __init__(self,
        method='taup',
        taup_model='ak135',
        FK_database=None,
        FK_model=None,
        **parameters):

        if not method:
            raise Exception('Undefined parameter: method')

        self.method = method
        self.taup_model = taup_model
        self.FK_database = FK_database
        self.FK_model = FK_model


        #
        # check parameters
        #
        if self.method == 'taup':
            assert self.taup_model is not None
            self._taup = TauPyModel(self.taup_model)

        elif self.method == 'FK_metadata':
            assert self.FK_database is not None
            assert exists(self.FK_database)
            if self.FK_model is None:
                self.FK_model = basename(self.FK_database)

        else:
            raise TypeError('Bad parameter: method')


    def __call__(self, data, greens, sources, progress_handle=Null(),
            set_attributes=False):

        #
        # check input arguments
        #
        _check(greens, self.method)

        if _type(sources.dims) == 'MomentTensor':
            sources = _to_array(sources)
            calculate_polarities = _polarities_mt

        elif _type(sources.dims) == 'Force':
            raise NotImplementedError
        else:
            raise TypeError

        #
        # collect observed polarities
        #
        if isinstance(data, mtuq.Dataset):
            observed = np.array([_get_polarity(stream) for stream in data])

        elif isinstance(data, list):
            observed = np.array(data)

        elif isinstance(data, np.ndarray):
            observed = data

        else:
            raise TypeError


        #
        # calculate predicted polarities
        #
        if self.method=='taup':
            takeoff_angles = _takeoff_angles_taup(
                self._taup, greens)

            predicted = calculate_polarities(
                sources, takeoff_angles, _collect_azimuths(greens))

        elif self.method=='FK_metadata':
            takeoff_angles = _takeoff_angles_FK(
                self.FK_database, greens)

            predicted = calculate_polarities(
                sources, takeoff_angles, _collect_azimuths(greens))


        #
        # evaluate misfit
        #
        values = np.abs(predicted - observed)/2.

        # mask unpicked
        mask = (observed != 0).astype(int)
        values = np.dot(values, mask)

        # returns a NumPy array of shape (len(sources), 1)
        return values.reshape(len(values), 1)


def _takeoff_angles_taup(taup, greens):
    """ Calculates takeoff angles from Tau-P model
    """

    takeoff_angles = np.zeros(len(greens))

    for _i, greens_tensor in enumerate(greens):

        depth_in_km = greens_tensor.origin.depth_in_m/1000.
        distance_in_deg = m_to_deg(greens_tensor.distance_in_m)

        arrivals = taup.get_travel_times(
            depth_in_km, distance_in_deg, phase_list=['p', 'P'])

        phases = []
        for arrival in arrivals:
            phases += [arrival.phase.name]

        if 'p' in phases:
            takeoff_angles[_i] = arrivals[phases.index('p')].takeoff_angle

        elif 'P' in phases:
            takeoff_angles[_i] = arrivals[phases.index('P')].takeoff_angle

        else:
            raise Exception

    return takeoff_angles


def _polarities_mt(mt_array, takeoff, azimuth):

        n1,n2 = mt_array.shape
        if n2!= 6:
            raise Exception('Inconsistent dimensions')

        n3,n4 = len(takeoff), len(azimuth)
        if n3!=n4:
            raise Exception('Inconsistent dimensions')

        # prepare arrays
        polarities = np.zeros((n1,n3))
        drc = np.empty((n1, 3))
        takeoff = np.deg2rad(takeoff)
        azimuth = np.deg2rad(azimuth)

        for _i, angles in enumerate(zip(takeoff, azimuth)):
            sth = np.sin(angles[0])
            cth = np.cos(angles[0])
            sphi = np.sin(angles[1])
            cphi = np.cos(angles[1])

            drc[:, 0] = sth*cphi
            drc[:, 1] = sth*sphi
            drc[:, 2] = cth

            # Aki & Richards 2ed, p. 108, eq. 4.88
            cth = mt_array[:, 0]*drc[:, 2]*drc[:, 2] +\
                  mt_array[:, 1]*drc[:, 0]*drc[:, 0] +\
                  mt_array[:, 2]*drc[:, 1]*drc[:, 1] +\
               2*(mt_array[:, 3]*drc[:, 0]*drc[:, 2] -
                  mt_array[:, 4]*drc[:, 1]*drc[:, 2] -
                  mt_array[:, 5]*drc[:, 0]*drc[:, 1])

            polarities[cth > 0, _i] = +1
            polarities[cth < 0, _i] = -1

        return polarities


def _polarities_force(force_array, takeoff_array, azimuth_array):
    raise NotImplementedError


def _collect_azimuths(greens):
    return np.array([stream.azimuth for stream in greens])


#
# input argument checking
#

def _model_type(greens):
    try:
        solver = greens[0].attrs.solver
    except:
        solver = 'Unknown'

    if solver in ('AxiSEM', 'FK', 'syngine'):
        model_type = '1D model'

    elif solver in ('SPECFEM3D', 'SPECFEM3D_GLOBE'):
        model_type = '3D model'

    else:
        model_type = 'Unknown model'

    return model_type


def _check(greens, method):
    return

    #model = _model_type(greens)
    #method = _method_type(method)

    #if model != method:
    #    print()
    #    print('  Possible inconsistency?')
    #    print('  Green''s functions are from: %s' % model)
    #    print('  Polarities are from: %s' % method)
    #    print()


