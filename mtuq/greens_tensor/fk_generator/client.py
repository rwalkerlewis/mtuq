"""
FK Generator Client for MTUQ Integration

This module provides a Client class that generates FK Green's functions
on-the-fly, integrating with MTUQ's existing infrastructure.
"""

import numpy as np
import obspy
from typing import Optional, Union, List

from mtuq.greens_tensor.FK import GreensTensor
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from obspy.geodetics import gps2dist_azimuth

from .velocity_model import VelocityModel, get_velocity_model
from .generator import FKGenerator, FKParameters


# Channel mapping for FK Green's functions
CHANNELS = [
    "TSS",
    "TDS",
    "REP",
    "RSS",
    "RDS",
    "RDD",
    "ZEP",
    "ZSS",
    "ZDS",
    "ZDD",
]


class Client(ClientBase):
    """
    FK Green's function generator client.

    This client generates FK Green's functions on-the-fly using a pure
    Python implementation of the frequency-wavenumber method.

    Unlike the FK_SAC client which reads pre-computed Green's functions,
    this client computes them dynamically based on the velocity model
    and source/receiver geometry.

    Example
    -------
    >>> from mtuq.greens_tensor.fk_generator import FKGeneratorClient, VelocityModel
    >>>
    >>> # Define or load a velocity model
    >>> model = VelocityModel(name='example')
    >>> model.add_layer(thickness=10.0, vp=6.3, vs=3.5, rho=2.786)
    >>> model.add_halfspace(vp=8.1, vs=4.7, rho=3.362)
    >>>
    >>> # Create client
    >>> client = Client(model=model)
    >>>
    >>> # Get Green's tensors for stations
    >>> greens = client.get_greens_tensors(stations, origins)

    Notes
    -----
    The Python implementation is optimized but may be slower than the
    compiled Fortran FK code for large numbers of stations/distances.
    For production use with many stations, consider pre-computing
    Green's functions using the Fortran code.
    """

    def __init__(
        self,
        model: Union[VelocityModel, str, None] = None,
        path_or_url: Optional[str] = None,
        include_mt: bool = True,
        include_force: bool = False,
        nt: int = 512,
        dt: float = 0.05,
        cache_greens: bool = True,
        **kwargs
    ):
        """
        Initialize the FK generator client.

        Parameters
        ----------
        model : VelocityModel, str, or None
            Velocity model instance, name of built-in model, or path to model file.
            If None and path_or_url is provided, attempts to load from path.
        path_or_url : str, optional
            Path to velocity model file (alternative to model parameter)
        include_mt : bool
            Include moment tensor sources (default True)
        include_force : bool
            Include force sources (default False, not yet implemented)
        nt : int
            Number of time points for Green's functions
        dt : float
            Sampling interval in seconds
        cache_greens : bool
            Cache computed Green's functions for reuse
        **kwargs
            Additional parameters passed to FKParameters
        """
        if include_force:
            raise NotImplementedError("Force sources not yet implemented in Python FK")

        # Load velocity model
        if model is not None:
            if isinstance(model, str):
                self.model = get_velocity_model(model)
            else:
                self.model = model
        elif path_or_url is not None:
            self.model = VelocityModel.from_file(path_or_url)
        else:
            raise ValueError("Must provide either 'model' or 'path_or_url'")

        self.model.validate()

        self.include_mt = include_mt
        self.include_force = include_force
        self.nt = nt
        self.dt = dt

        # Parameters for FK computation
        self.params = FKParameters(nt=nt, dt=dt, **kwargs)

        # Cache for computed Green's functions
        self.cache_greens = cache_greens
        self._cache = {}

    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """
        Generate Green's tensors for station-origin pairs.

        Parameters
        ----------
        stations : list
            List of Station objects
        origins : list
            List of Origin objects
        verbose : bool
            Print progress information

        Returns
        -------
        GreensTensorList
            List of Green's tensors, one per station-origin pair
        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)

    def _get_greens_tensor(self, station=None, origin=None):
        """
        Generate Green's tensor for a single station-origin pair.

        Parameters
        ----------
        station : Station
            MTUQ Station object
        origin : Origin
            MTUQ Origin object

        Returns
        -------
        GreensTensor
            FK Green's tensor
        """
        if station is None:
            raise ValueError("Missing station input argument")
        if origin is None:
            raise ValueError("Missing origin input argument")

        # Compute distance and azimuth
        distance_in_m, azimuth, _ = gps2dist_azimuth(
            origin.latitude, origin.longitude, station.latitude, station.longitude
        )
        distance_km = distance_in_m / 1000.0
        depth_km = origin.depth_in_m / 1000.0

        # Check cache
        cache_key = (round(distance_km, 1), round(depth_km, 1))

        if self.cache_greens and cache_key in self._cache:
            raw_traces = self._cache[cache_key]
        else:
            # Generate Green's functions
            raw_traces = self._compute_greens(distance_km, depth_km)

            if self.cache_greens:
                self._cache[cache_key] = raw_traces

        # Get time parameters from station
        t1_new = float(station.starttime)
        t2_new = float(station.endtime)
        dt_new = float(station.delta)

        # Create ObsPy traces
        traces = []

        for channel, data in raw_traces.items():
            trace = obspy.Trace(data=data.copy())
            trace.stats.channel = channel
            trace.stats._component = channel[0]

            # Original time parameters
            t1_old = float(origin.time) + raw_traces["t0"]
            t2_old = t1_old + (len(data) - 1) * raw_traces["dt"]
            dt_old = raw_traces["dt"]

            # Resample to match data
            data_new = resample(data, t1_old, t2_old, dt_old, t1_new, t2_new, dt_new)
            trace.data = data_new

            # Convert units: FK outputs 10^-20 cm/(dyne-cm)
            # MTUQ expects N^-1, conversion factor is 1e-15
            trace.data *= 1e-15

            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new

            traces.append(trace)

        tags = [
            f"model:{self.model.name}",
            "solver:FK_python",
        ]

        return GreensTensor(
            traces=traces,
            station=station,
            origin=origin,
            tags=tags,
            include_mt=self.include_mt,
            include_force=self.include_force,
        )

    def _compute_greens(self, distance_km: float, depth_km: float) -> dict:
        """
        Compute raw Green's functions for a given distance and depth.

        Parameters
        ----------
        distance_km : float
            Source-receiver distance in km
        depth_km : float
            Source depth in km

        Returns
        -------
        dict
            Dictionary mapping channel names to time series arrays
        """
        # Create generator for this depth
        generator = FKGenerator(
            self.model,
            source_depth_km=depth_km,
            source_type=2 if self.include_mt else 0,
        )

        # Compute Green's functions
        result = generator.compute(
            [distance_km], dt=self.dt, nt=self.nt, params=self.params
        )

        # Extract traces for this distance
        traces_dict = result["traces"][distance_km]
        traces_dict["dt"] = result["dt"]
        traces_dict["t0"] = result["t0"][0]

        return traces_dict

    def clear_cache(self):
        """Clear the Green's function cache."""
        self._cache = {}

    @property
    def cache_size(self):
        """Return number of cached Green's function sets."""
        return len(self._cache)


def open_db(model: Union[VelocityModel, str], **kwargs) -> Client:
    """
    Open an FK generator database.

    This is a convenience function for creating an FKGeneratorClient.

    Parameters
    ----------
    model : VelocityModel or str
        Velocity model or name/path
    **kwargs
        Additional arguments passed to Client

    Returns
    -------
    Client
        FK generator client

    Example
    -------
    >>> from mtuq.greens_tensor.fk_generator import open_db
    >>>
    >>> db = open_db('scak')
    >>> greens = db.get_greens_tensors(stations, origins)
    """
    return Client(model=model, **kwargs)
