"""
Velocity Model Classes for FK Green's Function Generation

This module provides classes for defining 1D layered velocity models
used in FK Green's function computation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union
import json


@dataclass
class Layer:
    """
    A single layer in a 1D velocity model.

    Parameters
    ----------
    thickness : float
        Layer thickness in km (0 for halfspace)
    vp : float
        P-wave velocity in km/s
    vs : float
        S-wave velocity in km/s
    rho : float
        Density in g/cm³
    qp : float
        P-wave quality factor
    qs : float
        S-wave quality factor
    """

    thickness: float
    vp: float
    vs: float
    rho: float
    qp: float = 1000.0
    qs: float = 500.0

    def __post_init__(self):
        """Validate layer parameters."""
        if self.vp <= 0:
            raise ValueError(f"P-wave velocity must be positive, got {self.vp}")
        if self.vs < 0:
            raise ValueError(f"S-wave velocity must be non-negative, got {self.vs}")
        if self.rho <= 0:
            raise ValueError(f"Density must be positive, got {self.rho}")
        if self.qp <= 0:
            raise ValueError(f"Qp must be positive, got {self.qp}")
        if self.qs <= 0:
            raise ValueError(f"Qs must be positive, got {self.qs}")

        # Handle fluid layers (vs=0)
        if self.vs < 1e-6:
            self.vs = 1e-6

    @property
    def mu(self) -> float:
        """Shear modulus: rho * vs²"""
        return self.rho * self.vs**2

    @property
    def lambda_(self) -> float:
        """First Lamé parameter: rho * (vp² - 2*vs²)"""
        return self.rho * (self.vp**2 - 2 * self.vs**2)

    @property
    def xi(self) -> float:
        """Ratio: vs²/vp²"""
        return (self.vs / self.vp) ** 2

    @property
    def is_halfspace(self) -> bool:
        """Check if this layer is a halfspace (thickness=0)."""
        return self.thickness < 1e-6


class VelocityModel:
    """
    1D layered velocity model for FK Green's function computation.

    The model consists of a stack of horizontal layers with constant
    properties, terminated by a halfspace at the bottom.

    Example
    -------
    >>> model = VelocityModel(name='example')
    >>> model.add_layer(thickness=10.0, vp=6.3, vs=3.5, rho=2.786)
    >>> model.add_layer(thickness=25.0, vp=6.3, vs=3.5, rho=2.786)
    >>> model.add_halfspace(vp=8.1, vs=4.7, rho=3.362)
    """

    def __init__(self, name: str = "custom", layers: Optional[List[Layer]] = None):
        """
        Initialize velocity model.

        Parameters
        ----------
        name : str
            Model name for identification
        layers : list of Layer, optional
            Pre-defined list of layers
        """
        self.name = name
        self._layers: List[Layer] = layers if layers is not None else []

    @property
    def layers(self) -> List[Layer]:
        """List of layers in the model."""
        return self._layers

    @property
    def n_layers(self) -> int:
        """Number of layers including halfspace."""
        return len(self._layers)

    def add_layer(
        self,
        thickness: float,
        vp: float,
        vs: float,
        rho: float,
        qp: float = 1000.0,
        qs: float = 500.0,
    ) -> "VelocityModel":
        """
        Add a layer to the model.

        Parameters
        ----------
        thickness : float
            Layer thickness in km
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in g/cm³
        qp : float
            P-wave quality factor
        qs : float
            S-wave quality factor

        Returns
        -------
        VelocityModel
            Self for method chaining
        """
        if thickness <= 0:
            raise ValueError(
                "Layer thickness must be positive. Use add_halfspace() for the bottom layer."
            )

        layer = Layer(thickness=thickness, vp=vp, vs=vs, rho=rho, qp=qp, qs=qs)
        self._layers.append(layer)
        return self

    def add_halfspace(
        self, vp: float, vs: float, rho: float, qp: float = 1000.0, qs: float = 500.0
    ) -> "VelocityModel":
        """
        Add the bottom halfspace.

        Parameters
        ----------
        vp : float
            P-wave velocity in km/s
        vs : float
            S-wave velocity in km/s
        rho : float
            Density in g/cm³
        qp : float
            P-wave quality factor
        qs : float
            S-wave quality factor

        Returns
        -------
        VelocityModel
            Self for method chaining
        """
        layer = Layer(thickness=0.0, vp=vp, vs=vs, rho=rho, qp=qp, qs=qs)
        self._layers.append(layer)
        return self

    def get_layer_at_depth(self, depth_km: float) -> tuple:
        """
        Find the layer containing a given depth.

        Parameters
        ----------
        depth_km : float
            Depth in km from surface

        Returns
        -------
        tuple
            (layer_index, Layer, depth_in_layer)
        """
        if depth_km < 0:
            raise ValueError(f"Depth must be non-negative, got {depth_km}")

        cumulative_depth = 0.0
        for i, layer in enumerate(self._layers):
            if layer.is_halfspace:
                return i, layer, depth_km - cumulative_depth

            if cumulative_depth + layer.thickness > depth_km:
                return i, layer, depth_km - cumulative_depth

            cumulative_depth += layer.thickness

        # If we get here, depth is in the halfspace
        return len(self._layers) - 1, self._layers[-1], depth_km - cumulative_depth

    def get_source_layer(self, source_depth_km: float) -> int:
        """
        Get the layer index (1-based) where the source is located.

        Following FK convention, the source is located at the TOP of the
        returned layer index.

        Parameters
        ----------
        source_depth_km : float
            Source depth in km

        Returns
        -------
        int
            Layer index (1-based) with source at top
        """
        cumulative_depth = 0.0
        for i, layer in enumerate(self._layers):
            if layer.is_halfspace:
                return i + 1  # 1-based

            next_depth = cumulative_depth + layer.thickness
            if abs(next_depth - source_depth_km) < 1e-6 or next_depth > source_depth_km:
                return i + 2  # 1-based, source at top of next layer

            cumulative_depth = next_depth

        return len(self._layers)  # In halfspace

    @property
    def max_velocity(self) -> float:
        """Maximum P-wave velocity in the model."""
        return max(layer.vp for layer in self._layers)

    @property
    def total_thickness(self) -> float:
        """Total thickness of finite layers (excluding halfspace)."""
        return sum(layer.thickness for layer in self._layers if not layer.is_halfspace)

    def depth_to_layer_top(self, layer_idx: int) -> float:
        """
        Get depth to the top of a layer (0-based index).

        Parameters
        ----------
        layer_idx : int
            Layer index (0-based)

        Returns
        -------
        float
            Depth to layer top in km
        """
        return sum(self._layers[i].thickness for i in range(layer_idx))

    def validate(self) -> bool:
        """
        Validate the velocity model.

        Returns
        -------
        bool
            True if model is valid

        Raises
        ------
        ValueError
            If model is invalid
        """
        if self.n_layers < 1:
            raise ValueError("Model must have at least one layer")

        if not self._layers[-1].is_halfspace:
            raise ValueError("Last layer must be a halfspace (thickness=0)")

        for i, layer in enumerate(self._layers[:-1]):
            if layer.is_halfspace:
                raise ValueError(
                    f"Only the last layer can be a halfspace, but layer {i} has thickness=0"
                )

        return True

    def to_arrays(self) -> dict:
        """
        Convert model to NumPy arrays for computation.

        Returns
        -------
        dict
            Dictionary containing arrays for d, vp, vs, rho, qp, qs, mu, xi
        """
        n = self.n_layers
        return {
            "d": np.array([l.thickness for l in self._layers]),
            "vp": np.array([l.vp for l in self._layers]),
            "vs": np.array([l.vs for l in self._layers]),
            "rho": np.array([l.rho for l in self._layers]),
            "qp": np.array([l.qp for l in self._layers]),
            "qs": np.array([l.qs for l in self._layers]),
            "mu": np.array([l.mu for l in self._layers]),
            "xi": np.array([l.xi for l in self._layers]),
        }

    def __repr__(self) -> str:
        lines = [f"VelocityModel(name='{self.name}', n_layers={self.n_layers})"]
        lines.append("-" * 70)
        lines.append(
            f"{'Layer':>6} {'Thickness':>10} {'Vp':>8} {'Vs':>8} {'Rho':>8} {'Qp':>8} {'Qs':>8}"
        )
        lines.append(
            f"{'':>6} {'(km)':>10} {'(km/s)':>8} {'(km/s)':>8} {'(g/cm³)':>8} {'':>8} {'':>8}"
        )
        lines.append("-" * 70)
        for i, layer in enumerate(self._layers):
            th_str = f"{layer.thickness:.3f}" if not layer.is_halfspace else "halfspace"
            lines.append(
                f"{i+1:>6} {th_str:>10} {layer.vp:>8.3f} {layer.vs:>8.3f} "
                f"{layer.rho:>8.3f} {layer.qp:>8.1f} {layer.qs:>8.1f}"
            )
        return "\n".join(lines)

    @classmethod
    def from_file(cls, filepath: str) -> "VelocityModel":
        """
        Load velocity model from file.

        Supports multiple formats:
        - JSON format
        - Simple text format (FK style)

        Parameters
        ----------
        filepath : str
            Path to model file

        Returns
        -------
        VelocityModel
            Loaded model
        """
        import os

        name = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, "r") as f:
            content = f.read().strip()

        # Try JSON format
        if content.startswith("{"):
            data = json.loads(content)
            return cls.from_dict(data)

        # Parse text format
        model = cls(name=name)
        lines = [
            l.strip()
            for l in content.split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]

        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                thickness = float(parts[0])
                vp = float(parts[1])
                vs = float(parts[2])
                rho = float(parts[3])
                qp = float(parts[4]) if len(parts) > 4 else 1000.0
                qs = float(parts[5]) if len(parts) > 5 else 500.0

                if thickness < 1e-6:
                    model.add_halfspace(vp=vp, vs=vs, rho=rho, qp=qp, qs=qs)
                else:
                    model.add_layer(
                        thickness=thickness, vp=vp, vs=vs, rho=rho, qp=qp, qs=qs
                    )

        return model

    @classmethod
    def from_dict(cls, data: dict) -> "VelocityModel":
        """
        Create model from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with 'name' and 'layers' keys

        Returns
        -------
        VelocityModel
            Created model
        """
        model = cls(name=data.get("name", "custom"))
        for layer_data in data.get("layers", []):
            layer = Layer(**layer_data)
            model._layers.append(layer)
        return model

    def to_dict(self) -> dict:
        """
        Convert model to dictionary.

        Returns
        -------
        dict
            Model as dictionary
        """
        return {
            "name": self.name,
            "layers": [
                {
                    "thickness": l.thickness,
                    "vp": l.vp,
                    "vs": l.vs,
                    "rho": l.rho,
                    "qp": l.qp,
                    "qs": l.qs,
                }
                for l in self._layers
            ],
        }

    def to_file(self, filepath: str, format: str = "json") -> None:
        """
        Save model to file.

        Parameters
        ----------
        filepath : str
            Output file path
        format : str
            'json' or 'text'
        """
        if format == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            with open(filepath, "w") as f:
                f.write(f"# Velocity model: {self.name}\n")
                f.write("# thickness(km)  Vp(km/s)  Vs(km/s)  rho(g/cm³)  Qp  Qs\n")
                for layer in self._layers:
                    f.write(
                        f"{layer.thickness:12.4f} {layer.vp:9.4f} {layer.vs:9.4f} "
                        f"{layer.rho:10.4f} {layer.qp:6.1f} {layer.qs:6.1f}\n"
                    )


# Pre-defined velocity models
def get_scak_model() -> VelocityModel:
    """
    Get the SCAK (Southern California - Alaska) velocity model.

    This is commonly used for regional seismology in Alaska.
    """
    model = VelocityModel(name="scak")
    model.add_layer(thickness=4.0, vp=5.3, vs=3.2, rho=2.4, qp=600, qs=300)
    model.add_layer(thickness=9.0, vp=5.6, vs=3.3, rho=2.67, qp=600, qs=300)
    model.add_layer(thickness=21.0, vp=6.2, vs=3.7, rho=2.8, qp=600, qs=300)
    model.add_layer(thickness=11.0, vp=7.2, vs=4.0, rho=3.1, qp=600, qs=300)
    model.add_halfspace(vp=7.9, vs=4.5, rho=3.38, qp=600, qs=300)
    return model


def get_ak135_model() -> VelocityModel:
    """
    Get a simplified AK135 velocity model (upper crust/mantle only).
    """
    model = VelocityModel(name="ak135")
    model.add_layer(thickness=20.0, vp=5.8, vs=3.46, rho=2.72, qp=600, qs=300)
    model.add_layer(thickness=15.0, vp=6.5, vs=3.85, rho=2.92, qp=600, qs=300)
    model.add_halfspace(vp=8.04, vs=4.48, rho=3.32, qp=1000, qs=500)
    return model


# Registry of built-in models
BUILTIN_MODELS = {
    "scak": get_scak_model,
    "ak135": get_ak135_model,
}


def get_velocity_model(name_or_path: str) -> VelocityModel:
    """
    Get a velocity model by name or load from file.

    Parameters
    ----------
    name_or_path : str
        Model name (for built-in models) or path to model file

    Returns
    -------
    VelocityModel
        Velocity model
    """
    if name_or_path in BUILTIN_MODELS:
        return BUILTIN_MODELS[name_or_path]()

    import os

    if os.path.exists(name_or_path):
        return VelocityModel.from_file(name_or_path)

    raise ValueError(
        f"Unknown model '{name_or_path}'. "
        f"Available built-in models: {list(BUILTIN_MODELS.keys())}"
    )
