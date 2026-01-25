"""
FK Green's Function Generator

Main class for computing Green's functions using the frequency-wavenumber
method in horizontally layered media.
"""

import numpy as np
from typing import Optional, Union, List, Tuple
import warnings
from dataclasses import dataclass

from .velocity_model import VelocityModel, get_velocity_model
from .haskell import compute_source_coefficients, EPSILON, PI, PI2


@dataclass
class FKParameters:
    """
    Parameters for FK computation.

    Parameters
    ----------
    nt : int
        Number of time points (must be power of 2)
    dt : float
        Sampling interval in seconds
    sigma : float
        Damping factor (in 1/trace_length, typically 2-3)
    taper : float
        Low-pass taper factor (0-1)
    nb : int
        Number of samples before first arrival
    smth : int
        Smoothing factor (oversampling, power of 2)
    pmin : float
        Minimum slowness (in 1/vs at source)
    pmax : float
        Maximum slowness (in 1/vs at source)
    dk : float
        Wavenumber sampling (in pi/max(x,hs))
    kmax : float
        Maximum wavenumber at omega=0 (in 1/hs)
    wc1 : int
        High-pass filter corner 1
    wc2 : int
        High-pass filter corner 2
    """

    nt: int = 512
    dt: float = 0.1
    sigma: float = 2.0
    taper: float = 0.5
    nb: int = 25
    smth: int = 1
    pmin: float = 0.0
    pmax: float = 1.0
    dk: float = 0.3
    kmax: float = 15.0
    wc1: int = 1
    wc2: int = 1


class FKGenerator:
    """
    Generator for FK Green's functions in horizontally layered media.

    This class computes elastic Green's functions using the frequency-wavenumber
    integration method with Thompson-Haskell propagator matrices.

    Example
    -------
    >>> from mtuq.greens_tensor.fk_generator import FKGenerator, VelocityModel
    >>>
    >>> model = VelocityModel(name='simple')
    >>> model.add_layer(thickness=35.0, vp=6.0, vs=3.5, rho=2.7)
    >>> model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)
    >>>
    >>> gen = FKGenerator(model, source_depth_km=10.0)
    >>> greens = gen.compute([50, 100, 150], dt=0.1, nt=512)

    References
    ----------
    Zhu & Rivera (2002), Geophys. J. Int.
    Haskell (1964), Bull. Seismol. Soc. Am.
    """

    def __init__(
        self,
        model: Union[VelocityModel, str],
        source_depth_km: float,
        receiver_depth_km: float = 0.0,
        source_type: int = 2,
        updn: int = 0,
    ):
        """
        Initialize FK generator.

        Parameters
        ----------
        model : VelocityModel or str
            Velocity model instance or name of built-in model
        source_depth_km : float
            Source depth in km
        receiver_depth_km : float
            Receiver depth in km (default 0 = surface)
        source_type : int
            0 = explosion, 1 = single force, 2 = double couple
        updn : int
            0 = all waves, 1 = down-going only, -1 = up-going only
        """
        # Load model if string
        if isinstance(model, str):
            self.model = get_velocity_model(model)
        else:
            self.model = model

        self.model.validate()

        self.source_depth_km = source_depth_km
        self.receiver_depth_km = receiver_depth_km
        self.source_type = source_type
        self.updn = updn

        # Determine source and receiver layers
        self._setup_geometry()

        # Pre-compute source coefficients
        self._setup_source()

    def _setup_geometry(self):
        """Setup source and receiver geometry."""
        # Find source layer
        src_idx, src_layer, _ = self.model.get_layer_at_depth(self.source_depth_km)
        self.src_layer = src_idx + 1  # 1-based

        # Find receiver layer
        rcv_idx, rcv_layer, _ = self.model.get_layer_at_depth(self.receiver_depth_km)
        self.rcv_layer = rcv_idx + 1  # 1-based

        # Flip model if receiver below source
        self.flip = 1
        if self.rcv_layer > self.src_layer:
            self.flip = -1
            n = self.model.n_layers
            self.src_layer = n - self.src_layer + 2
            self.rcv_layer = n - self.rcv_layer + 2

        # Compute source-receiver separation
        self.hs = abs(self.source_depth_km - self.receiver_depth_km)
        if self.hs < EPSILON:
            self.hs = self.source_depth_km  # Use depth from surface

    def _setup_source(self):
        """Pre-compute source coefficients."""
        # Get properties at source layer
        src_idx = (
            self.src_layer - 1
            if self.flip > 0
            else self.model.n_layers - self.src_layer
        )
        src_layer = self.model.layers[src_idx]

        self.vs_source = src_layer.vs
        self.si = compute_source_coefficients(
            self.source_type, src_layer.xi, src_layer.mu, self.flip
        )

    def compute(
        self,
        distances_km: Union[List[float], np.ndarray],
        dt: float = 0.1,
        nt: int = 512,
        t0: Optional[Union[float, List[float]]] = None,
        params: Optional[FKParameters] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Compute Green's functions at specified distances.

        Parameters
        ----------
        distances_km : list or array
            Distances in km
        dt : float
            Sampling interval in seconds
        nt : int
            Number of time points (should be power of 2)
        t0 : float or list, optional
            Start time(s) relative to origin time. If None, computed automatically.
        params : FKParameters, optional
            Advanced computation parameters
        verbose : bool
            Print progress information

        Returns
        -------
        dict
            Dictionary with keys:
            - 'traces': dict mapping distance to dict of component time series
            - 'dt': sampling interval
            - 'nt': number of points
            - 't0': start times
            - 'distances': distance array
            - 'model': velocity model name
            - 'source_depth': source depth
        """
        distances_km = np.atleast_1d(distances_km).astype(float)
        nx = len(distances_km)

        # Setup parameters
        if params is None:
            params = FKParameters(nt=nt, dt=dt)
        else:
            params.nt = nt
            params.dt = dt

        # Validate nt is power of 2
        if nt & (nt - 1) != 0:
            nt_new = 2 ** int(np.ceil(np.log2(nt)))
            warnings.warn(f"nt={nt} is not a power of 2, using nt={nt_new}")
            params.nt = nt_new
            nt = nt_new

        # Compute t0 if not specified
        if t0 is None:
            # Estimate from P-wave arrival
            t0 = np.zeros(nx)
            for i, x in enumerate(distances_km):
                dist = np.sqrt(x**2 + self.hs**2)
                t0[i] = dist / self.model.max_velocity - params.nb * dt
                t0[i] = max(0, t0[i])
        else:
            t0 = np.atleast_1d(t0)
            if len(t0) == 1:
                t0 = np.full(nx, t0[0])

        # Compute numerical parameters
        xmax = max(np.max(distances_km), self.hs)
        vmax = self.model.max_velocity

        # Wavenumber parameters
        dk = params.dk * PI / xmax
        kc = params.kmax / self.hs
        pmin = params.pmin / self.vs_source
        pmax = params.pmax / self.vs_source

        # Frequency parameters
        nfft = nt
        nfft2 = nfft // 2
        dw = PI2 / (nfft * dt)
        sigma = params.sigma * dw / PI2

        # Taper parameters
        wc = max(1, int(nfft2 * (1.0 - params.taper)))
        taper_slope = PI / (nfft2 - wc + 1) if nfft2 > wc else PI
        wc1 = min(params.wc1, params.wc2)
        wc2 = min(params.wc2, wc)

        if verbose:
            print(f"FK computation: nt={nt}, dt={dt}, nfft={nfft}")
            print(f"  dk={dk:.5f}, kmax={kc:.2f}, pmax={pmax:.4f}")
            print(f"  frequencies: {wc1} to {nfft2}")

        # Number of components
        n_com = 3 + 3 * self.source_type

        # Initialize sums
        sums = np.zeros((nx, n_com, nfft2), dtype=np.complex128)

        # Model arrays for computation
        model_arrays = self.model.to_arrays()

        # Import kernel functions
        from .kernel import integrate_wavenumber

        # Frequency loop
        for j in range(wc1, nfft2 + 1):
            omega_real = (j - 1) * dw
            omega = complex(omega_real, -sigma)

            if omega_real < EPSILON:
                continue

            # Wavenumber limits for this frequency
            k_min = omega_real * pmin + 0.5 * dk
            k_max = np.sqrt(kc**2 + (pmax * omega_real) ** 2)

            # Integrate over wavenumber
            freq_sums = integrate_wavenumber(
                omega,
                model_arrays,
                self.model.n_layers,
                self.src_layer,
                self.rcv_layer,
                self.source_type,
                self.updn,
                self.flip,
                self.si,
                distances_km,
                k_min,
                k_max,
                dk,
            )

            # Apply taper and phase shift
            for ix in range(nx):
                phi = omega_real * t0[ix]
                filt = dk / PI2

                if j > wc:
                    filt *= 0.5 * (1.0 + np.cos((j - wc) * taper_slope))
                if j < wc2:
                    filt *= 0.5 * (1.0 + np.cos((wc2 - j) * PI / (wc2 - wc1)))

                att = filt * np.exp(1j * phi)

                for l in range(n_com):
                    sums[ix, l, j - 1] = freq_sums[ix, l] * att

            if verbose and j % 100 == 0:
                print(f"  frequency {j}/{nfft2}")

        # Inverse FFT
        smth = params.smth
        dt_out = dt / smth
        nfft_out = smth * nfft
        nfft3 = nfft_out // 2
        dfac = np.exp(sigma * dt_out)

        # Store results
        traces = {}

        # Component naming based on source type
        if self.source_type == 2:  # Double couple
            components = ["ZDD", "RDD", "TDD", "ZDS", "RDS", "TDS", "ZSS", "RSS", "TSS"]
            # Map to FK file extensions
            ext_map = {
                "ZDD": "0",
                "ZDS": "3",
                "ZSS": "6",
                "ZEP": "a",
                "RDD": "1",
                "RDS": "4",
                "RSS": "7",
                "REP": "b",
                "TDD": "2",
                "TDS": "5",
                "TSS": "8",
            }
        elif self.source_type == 0:  # Explosion
            components = ["ZEP", "REP", "TEP"]
        else:  # Single force
            components = ["ZSF", "RSF", "TSF", "ZDF", "RDF", "TDF"]

        for ix, x in enumerate(distances_km):
            traces[x] = {}

            for l in range(n_com):
                # Prepare frequency-domain data
                data_freq = np.zeros(nfft3, dtype=np.complex128)
                data_freq[:nfft2] = sums[ix, l, :]

                # IFFT
                data_time = np.fft.irfft(data_freq, n=nfft_out)

                # Remove damping
                z = np.exp(sigma * t0[ix])
                for i in range(len(data_time)):
                    data_time[i] *= z
                    z *= dfac

                # Store with proper component name
                if l < len(components):
                    traces[x][components[l]] = data_time.real

        # Also compute explosion components for moment tensor
        # ZEP = ZDD + ZSS (isotropic component)
        if self.source_type == 2:
            for ix, x in enumerate(distances_km):
                # EP components are combinations for isotropic source
                traces[x]["ZEP"] = traces[x]["ZDD"] / 3.0 + traces[x]["ZSS"] * 0
                traces[x]["REP"] = traces[x]["RDD"] / 3.0 + traces[x]["RSS"] * 0

        return {
            "traces": traces,
            "dt": dt_out,
            "nt": nfft_out,
            "t0": t0,
            "distances": distances_km,
            "model": self.model.name,
            "source_depth": self.source_depth_km,
            "components": components,
        }

    def compute_for_station(
        self,
        station_distance_km: float,
        station_azimuth_deg: float,
        dt: float = 0.1,
        nt: int = 512,
        t0: Optional[float] = None,
    ) -> dict:
        """
        Compute Green's functions for a single station.

        Parameters
        ----------
        station_distance_km : float
            Source-station distance in km
        station_azimuth_deg : float
            Station azimuth in degrees
        dt : float
            Sampling interval
        nt : int
            Number of samples
        t0 : float, optional
            Start time

        Returns
        -------
        dict
            Green's function traces for this station
        """
        result = self.compute([station_distance_km], dt=dt, nt=nt, t0=t0)
        traces = result["traces"][station_distance_km]

        return {
            "traces": traces,
            "distance": station_distance_km,
            "azimuth": station_azimuth_deg,
            "dt": result["dt"],
            "nt": result["nt"],
            "t0": result["t0"][0],
        }

    def to_mtuq_greens_tensor(self, station, origin, traces_dict: dict):
        """
        Convert computed Green's functions to MTUQ GreensTensor format.

        Parameters
        ----------
        station : Station
            MTUQ Station object
        origin : Origin
            MTUQ Origin object
        traces_dict : dict
            Output from compute() or compute_for_station()

        Returns
        -------
        GreensTensor
            MTUQ-compatible Green's tensor
        """
        import obspy
        from mtuq.greens_tensor.FK import GreensTensor

        traces = []
        dt = traces_dict["dt"]
        t0 = traces_dict.get("t0", 0)
        if isinstance(t0, np.ndarray):
            t0 = t0[0]

        channel_data = traces_dict.get("traces", traces_dict)

        # Map component names to MTUQ channels
        for channel_name, data in channel_data.items():
            if isinstance(data, np.ndarray):
                trace = obspy.Trace(data=data.astype(np.float32))
                trace.stats.channel = channel_name
                trace.stats._component = channel_name[0]  # Z, R, or T
                trace.stats.starttime = float(origin.time) + t0
                trace.stats.delta = dt

                # Convert units: computed in 10^-20 cm/(dyne-cm)
                # MTUQ expects N^-1, conversion factor is 1e-15
                trace.data *= 1e-15

                traces.append(trace)

        return GreensTensor(
            traces=traces,
            station=station,
            origin=origin,
            tags=[f"model:{self.model.name}", "solver:FK_python"],
            include_mt=(self.source_type == 2),
            include_force=(self.source_type == 1),
        )


def generate_greens_functions(
    model: Union[VelocityModel, str],
    source_depth_km: float,
    distances_km: Union[List[float], np.ndarray],
    dt: float = 0.1,
    nt: int = 512,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Convenience function to generate Green's functions.

    Parameters
    ----------
    model : VelocityModel or str
        Velocity model
    source_depth_km : float
        Source depth in km
    distances_km : list or array
        Distances in km
    dt : float
        Sampling interval in seconds
    nt : int
        Number of time points
    output_dir : str, optional
        Directory to save SAC files (FK format)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Computed Green's functions
    """
    gen = FKGenerator(model, source_depth_km)
    result = gen.compute(distances_km, dt=dt, nt=nt, verbose=verbose)

    if output_dir is not None:
        _save_to_sac(result, output_dir)

    return result


def _save_to_sac(result: dict, output_dir: str):
    """Save results in FK-compatible SAC format."""
    import os
    import obspy

    os.makedirs(output_dir, exist_ok=True)

    # Extension mapping for FK format
    ext_map = {
        "ZDD": "0",
        "ZDS": "3",
        "ZSS": "6",
        "ZEP": "a",
        "RDD": "1",
        "RDS": "4",
        "RSS": "7",
        "REP": "b",
        "TDD": "2",
        "TDS": "5",
        "TSS": "8",
    }

    dt = result["dt"]

    for dist, traces in result["traces"].items():
        dist_int = int(dist)
        for comp, data in traces.items():
            if comp in ext_map:
                ext = ext_map[comp]
                filename = os.path.join(output_dir, f"{dist_int}.grn.{ext}")

                trace = obspy.Trace(data=data.astype(np.float32))
                trace.stats.delta = dt
                trace.stats.starttime = 0

                trace.write(filename, format="SAC")
