"""
FK Green's Function Generator

A pure Python implementation of the FK (frequency-wavenumber) method for
computing Green's functions in horizontally layered media.

This module implements the Thompson-Haskell propagator matrix method
combined with frequency-wavenumber integration, following the algorithm
described by Zhu & Rivera (2002).

References:
    Haskell (1964), BSSA - Original propagator matrix method
    Wang & Herrmann (1980), BSSA - Compound matrix formulation
    Zhu & Rivera (2002), GJI - FK implementation details

Example usage:
    >>> from mtuq.greens_tensor.fk_generator import FKGenerator, VelocityModel
    >>>
    >>> # Define velocity model
    >>> model = VelocityModel()
    >>> model.add_layer(thickness=10.0, vp=6.3, vs=3.5, rho=2.786, qp=1000, qs=500)
    >>> model.add_layer(thickness=25.0, vp=6.3, vs=3.5, rho=2.786, qp=1000, qs=500)
    >>> model.add_halfspace(vp=8.1, vs=4.7, rho=3.362, qp=1600, qs=800)
    >>>
    >>> # Generate Green's functions
    >>> generator = FKGenerator(model, source_depth_km=15.0)
    >>> greens = generator.compute(distances_km=[100, 200, 300], dt=0.1, nt=512)

"""

from .velocity_model import VelocityModel
from .generator import FKGenerator
from .client import Client as FKGeneratorClient

__all__ = ["VelocityModel", "FKGenerator", "FKGeneratorClient"]
