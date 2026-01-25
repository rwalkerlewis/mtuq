"""
Example: Generate FK Green's Functions with Python Implementation

This example demonstrates how to use the Python FK generator to compute
Green's functions for a horizontally layered velocity model, and compares
with pre-computed FK Green's functions.

This serves as both an example of usage and a validation of the implementation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from obspy import read

# Add workspace to path
sys.path.insert(0, "/workspace")

# Import MTUQ components
from mtuq.util import fullpath
from mtuq.greens_tensor.fk_generator import (
    VelocityModel,
    FKGenerator,
    FKGeneratorClient,
)


def create_scak_model():
    """
    Create the SCAK velocity model used in the benchmark tests.

    This model is used in the example event 20090407201255351.
    """
    model = VelocityModel(name="scak")

    # SCAK velocity model (Southern California - Alaska)
    # These values are estimates - the exact values used for the benchmark
    # would need to be obtained from the original FK computation
    model.add_layer(thickness=4.0, vp=5.3, vs=3.2, rho=2.4, qp=600, qs=300)
    model.add_layer(thickness=9.0, vp=5.6, vs=3.3, rho=2.67, qp=600, qs=300)
    model.add_layer(thickness=21.0, vp=6.2, vs=3.7, rho=2.8, qp=600, qs=300)
    model.add_layer(thickness=11.0, vp=7.2, vs=4.0, rho=3.1, qp=600, qs=300)
    model.add_halfspace(vp=7.9, vs=4.5, rho=3.38, qp=600, qs=300)

    return model


def example_basic_usage():
    """
    Demonstrates basic usage of the FK generator.
    """
    print("=" * 60)
    print("Example 1: Basic FK Green's Function Generation")
    print("=" * 60)

    # Create a simple velocity model
    model = VelocityModel(name="simple_model")
    model.add_layer(thickness=10.0, vp=6.0, vs=3.5, rho=2.7, qp=600, qs=300)
    model.add_layer(thickness=20.0, vp=6.5, vs=3.8, rho=2.9, qp=600, qs=300)
    model.add_halfspace(vp=8.0, vs=4.5, rho=3.3, qp=1000, qs=500)

    print("\nVelocity Model:")
    print(model)

    # Create FK generator
    source_depth = 15.0  # km
    generator = FKGenerator(model, source_depth_km=source_depth)

    print(f"\nSource depth: {source_depth} km")
    print(f"Source layer: {generator.src_layer}")

    # Compute Green's functions for a range of distances
    distances = [50.0, 100.0, 150.0]  # km

    print(f"\nComputing Green's functions for distances: {distances} km")
    result = generator.compute(distances, dt=0.5, nt=128, verbose=True)

    print(f"\nResults:")
    print(f"  Number of time points: {result['nt']}")
    print(f"  Sampling interval: {result['dt']} s")
    print(f"  Components: {result['components']}")

    for dist in distances:
        traces = result["traces"][dist]
        print(f"\n  Distance {dist} km:")
        for comp in ["ZDD", "ZDS", "ZSS"]:
            if comp in traces:
                data = traces[comp]
                print(f"    {comp}: max amplitude = {np.max(np.abs(data)):.6e}")

    return result


def example_with_client():
    """
    Demonstrates using the FK generator through the MTUQ client interface.
    """
    print("\n" + "=" * 60)
    print("Example 2: Using FKGeneratorClient")
    print("=" * 60)

    # Create client with SCAK model
    model = create_scak_model()
    client = FKGeneratorClient(model=model, nt=256, dt=0.1)

    print(f"\nClient created with model: {client.model.name}")
    print(f"  Number of layers: {client.model.n_layers}")
    print(f"  Cache enabled: {client.cache_greens}")

    # The client can be used with MTUQ's get_greens_tensors() method
    # when stations and origins are provided

    return client


def example_compare_with_precomputed():
    """
    Compare Python FK with pre-computed FK Green's functions.

    Note: This comparison will show differences because:
    1. The exact velocity model parameters may differ
    2. Numerical parameters (dk, kmax, etc.) may differ
    3. The Python implementation uses a slightly different algorithm

    The comparison is mainly to verify the implementation produces
    physically reasonable results.
    """
    print("\n" + "=" * 60)
    print("Example 3: Comparison with Pre-computed FK")
    print("=" * 60)

    # Path to pre-computed Green's functions
    path_greens = fullpath("data/tests/benchmark_cap/greens/scak/scak_34")

    if not os.path.exists(path_greens):
        print(f"Pre-computed Green's functions not found at {path_greens}")
        print("Skipping comparison.")
        return None

    # Read a pre-computed trace
    distance = 100  # km

    try:
        # Read ZDD component (extension .0)
        trace_fk = read(f"{path_greens}/{distance}.grn.0", format="sac")[0]
        print(f"\nPre-computed FK (distance={distance} km, component=ZDD):")
        print(f"  Number of points: {len(trace_fk.data)}")
        print(f"  Sampling interval: {trace_fk.stats.delta} s")
        print(f"  Max amplitude: {np.max(np.abs(trace_fk.data)):.6e}")

        # Create model and compute Python FK
        model = create_scak_model()
        generator = FKGenerator(model, source_depth_km=34.0)  # 34 km from scak_34

        result = generator.compute(
            [float(distance)], dt=trace_fk.stats.delta, nt=len(trace_fk.data)
        )

        if "ZDD" in result["traces"][float(distance)]:
            python_data = result["traces"][float(distance)]["ZDD"]

            print(f"\nPython FK (distance={distance} km, component=ZDD):")
            print(f"  Number of points: {len(python_data)}")
            print(f"  Max amplitude: {np.max(np.abs(python_data)):.6e}")

            # Plot comparison (if matplotlib available)
            try:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))

                t_fk = np.arange(len(trace_fk.data)) * trace_fk.stats.delta
                t_py = np.arange(len(python_data)) * result["dt"]

                axes[0].plot(t_fk, trace_fk.data, "b-", label="Pre-computed FK")
                axes[0].set_title(
                    f"Pre-computed FK Green's Function (ZDD, {distance} km)"
                )
                axes[0].set_xlabel("Time (s)")
                axes[0].set_ylabel("Amplitude")
                axes[0].legend()

                axes[1].plot(t_py, python_data, "r-", label="Python FK")
                axes[1].set_title(f"Python FK Green's Function (ZDD, {distance} km)")
                axes[1].set_xlabel("Time (s)")
                axes[1].set_ylabel("Amplitude")
                axes[1].legend()

                plt.tight_layout()
                output_path = "/workspace/fk_comparison.png"
                plt.savefig(output_path, dpi=100)
                print(f"\nComparison plot saved to: {output_path}")
                plt.close()

            except Exception as e:
                print(f"Could not create plot: {e}")

    except Exception as e:
        print(f"Error reading pre-computed traces: {e}")
        return None


def example_custom_parameters():
    """
    Demonstrates customizing FK computation parameters.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Parameters")
    print("=" * 60)

    from mtuq.greens_tensor.fk_generator.generator import FKParameters

    # Create custom parameters
    params = FKParameters(
        sigma=3.0,  # Higher damping
        taper=0.3,  # Different taper
        pmin=0.0,  # Minimum slowness
        pmax=1.2,  # Maximum slowness
        dk=0.2,  # Finer wavenumber sampling
        kmax=20.0,  # Higher kmax for shallow sources
    )

    print("\nCustom parameters:")
    print(f"  sigma: {params.sigma}")
    print(f"  taper: {params.taper}")
    print(f"  pmin: {params.pmin}")
    print(f"  pmax: {params.pmax}")
    print(f"  dk: {params.dk}")
    print(f"  kmax: {params.kmax}")

    # Create simple model
    model = VelocityModel(name="custom")
    model.add_layer(thickness=35.0, vp=6.0, vs=3.5, rho=2.7)
    model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

    generator = FKGenerator(model, source_depth_km=10.0)
    result = generator.compute([100.0], dt=0.2, nt=256, params=params)

    print(f"\nComputation completed:")
    print(f"  Distance: 100 km")
    print(f"  Components available: {list(result['traces'][100.0].keys())}")

    return result


def example_velocity_model_io():
    """
    Demonstrates saving and loading velocity models.
    """
    print("\n" + "=" * 60)
    print("Example 5: Velocity Model I/O")
    print("=" * 60)

    # Create a model
    model = create_scak_model()

    # Save to JSON
    json_path = "/workspace/scak_model.json"
    model.to_file(json_path, format="json")
    print(f"\nModel saved to JSON: {json_path}")

    # Save to text format
    txt_path = "/workspace/scak_model.txt"
    model.to_file(txt_path, format="text")
    print(f"Model saved to text: {txt_path}")

    # Load back from JSON
    loaded_model = VelocityModel.from_file(json_path)
    print(f"\nLoaded model from JSON:")
    print(f"  Name: {loaded_model.name}")
    print(f"  Layers: {loaded_model.n_layers}")

    return model


if __name__ == "__main__":
    print("FK Green's Function Generator - Examples")
    print("=" * 60)

    # Run examples
    example_basic_usage()
    example_with_client()
    example_compare_with_precomputed()
    example_custom_parameters()
    example_velocity_model_io()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
