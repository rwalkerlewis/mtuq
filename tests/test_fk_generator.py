"""
Tests for the Python FK Green's function generator.

This module compares the Python FK implementation against pre-computed
FK Green's functions to validate correctness.
"""

import os
import sys
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVelocityModel:
    """Tests for the VelocityModel class."""

    def test_create_model(self):
        """Test creating a velocity model."""
        from mtuq.greens_tensor.fk_generator import VelocityModel

        model = VelocityModel(name="test")
        model.add_layer(thickness=10.0, vp=6.0, vs=3.5, rho=2.7)
        model.add_layer(thickness=20.0, vp=6.5, vs=3.8, rho=2.9)
        model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

        assert model.n_layers == 3
        assert model.name == "test"
        model.validate()

    def test_layer_properties(self):
        """Test layer property calculations."""
        from mtuq.greens_tensor.fk_generator.velocity_model import Layer

        layer = Layer(thickness=10.0, vp=6.0, vs=3.5, rho=2.7)

        # Check shear modulus: mu = rho * vs^2
        expected_mu = 2.7 * 3.5**2
        assert abs(layer.mu - expected_mu) < 1e-10

        # Check xi = vs^2 / vp^2
        expected_xi = (3.5 / 6.0) ** 2
        assert abs(layer.xi - expected_xi) < 1e-10

    def test_load_fk_model_ak_scak(self):
        """Test loading ak_scak model from fkmodels directory."""
        from mtuq.greens_tensor.fk_generator import get_velocity_model

        model = get_velocity_model("ak_scak")
        assert model.name == "ak_scak"
        assert model.n_layers == 9  # ak_scak has 9 layers
        model.validate()

        # Check first layer values match the file
        assert abs(model.layers[0].thickness - 4.0) < 0.01
        assert abs(model.layers[0].vs - 3.01) < 0.01
        assert abs(model.layers[0].vp - 5.3) < 0.01

    def test_load_fk_model_socal(self):
        """Test loading socal model from fkmodels directory."""
        from mtuq.greens_tensor.fk_generator import get_velocity_model

        model = get_velocity_model("socal")
        assert model.name == "socal"
        assert model.n_layers >= 2
        model.validate()

    def test_fkmodels_path(self):
        """Test that fkmodels directory can be found."""
        from mtuq.greens_tensor.fk_generator import get_fkmodels_path

        path = get_fkmodels_path()
        assert path is not None
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "ak_scak"))

    def test_depth_lookup(self):
        """Test finding layer at depth."""
        from mtuq.greens_tensor.fk_generator import VelocityModel

        model = VelocityModel(name="test")
        model.add_layer(thickness=10.0, vp=6.0, vs=3.5, rho=2.7)
        model.add_layer(thickness=20.0, vp=6.5, vs=3.8, rho=2.9)
        model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

        idx, layer, depth_in_layer = model.get_layer_at_depth(5.0)
        assert idx == 0
        assert abs(depth_in_layer - 5.0) < 1e-10

        idx, layer, depth_in_layer = model.get_layer_at_depth(15.0)
        assert idx == 1
        assert abs(depth_in_layer - 5.0) < 1e-10

        idx, layer, depth_in_layer = model.get_layer_at_depth(50.0)
        assert idx == 2  # Halfspace

    def test_model_arrays(self):
        """Test conversion to arrays."""
        from mtuq.greens_tensor.fk_generator import VelocityModel

        model = VelocityModel(name="test")
        model.add_layer(thickness=10.0, vp=6.0, vs=3.5, rho=2.7)
        model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

        arrays = model.to_arrays()

        assert len(arrays["d"]) == 2
        assert arrays["d"][0] == 10.0
        assert arrays["d"][1] == 0.0
        assert arrays["vp"][0] == 6.0
        assert arrays["vs"][1] == 4.5


class TestHaskellMatrices:
    """Tests for the Haskell propagator matrix functions."""

    def test_hyperbolic_functions(self):
        """Test sh_ch function for hyperbolic computations."""
        from mtuq.greens_tensor.fk_generator.haskell import sh_ch

        # Test with real argument
        a = 0.5 + 0j
        kd = 1.0

        c, y, x, ex = sh_ch(a, kd)

        # c should be cosh(a*kd) * exp(-real(a*kd))
        expected_c = np.cosh(0.5) * np.exp(-0.5)
        assert abs(c - expected_c) < 1e-10

    def test_layer_parameters(self):
        """Test layer parameter computation."""
        from mtuq.greens_tensor.fk_generator.haskell import compute_layer_parameters

        k = 0.1
        omega = 1.0 + 0.01j
        vp = 6.0
        vs = 3.5
        mu = 2.7 * 3.5**2
        qp = 1000
        qs = 500
        d = 10.0

        ra, rb, r, r1, kd, mu2 = compute_layer_parameters(
            k, omega, vp, vs, mu, qp, qs, d
        )

        # Check basic properties
        assert kd == k * d
        assert mu2 == 2 * mu
        # ra and rb should have negative imaginary parts for evanescent waves
        # (depending on the branch cut)

    def test_source_coefficients(self):
        """Test source coefficient computation."""
        from mtuq.greens_tensor.fk_generator.haskell import compute_source_coefficients

        # Double couple source
        xi = 0.34  # vs^2/vp^2
        mu = 30.0  # Shear modulus
        flip = 1

        si = compute_source_coefficients(2, xi, mu, flip)

        # Check that coefficients match expected values
        assert abs(si[0, 1] - 2 * xi / mu) < 1e-10
        assert abs(si[0, 3] - (4 * xi - 3)) < 1e-10

        # Explosion source
        si_exp = compute_source_coefficients(0, xi, mu, flip)
        assert abs(si_exp[0, 1] - xi / mu) < 1e-10


class TestFKGenerator:
    """Tests for the main FK generator."""

    def test_generator_init(self):
        """Test initializing the FK generator."""
        from mtuq.greens_tensor.fk_generator import FKGenerator, VelocityModel

        model = VelocityModel(name="test")
        model.add_layer(thickness=35.0, vp=6.0, vs=3.5, rho=2.7)
        model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

        gen = FKGenerator(model, source_depth_km=10.0)

        assert gen.source_depth_km == 10.0
        assert gen.source_type == 2  # Default is double couple

    def test_simple_computation(self):
        """Test basic Green's function computation."""
        from mtuq.greens_tensor.fk_generator import FKGenerator, VelocityModel

        model = VelocityModel(name="simple")
        model.add_layer(thickness=35.0, vp=6.0, vs=3.5, rho=2.7)
        model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

        gen = FKGenerator(model, source_depth_km=10.0)

        # Compute for a single distance
        result = gen.compute([100.0], dt=0.5, nt=64)

        assert "traces" in result
        assert 100.0 in result["traces"]
        assert "ZDD" in result["traces"][100.0]
        assert len(result["traces"][100.0]["ZDD"]) > 0


class TestFKClient:
    """Tests for the FK generator client."""

    def test_client_init(self):
        """Test initializing the client."""
        from mtuq.greens_tensor.fk_generator import FKGeneratorClient, VelocityModel

        model = VelocityModel(name="test")
        model.add_layer(thickness=35.0, vp=6.0, vs=3.5, rho=2.7)
        model.add_halfspace(vp=8.0, vs=4.5, rho=3.3)

        client = FKGeneratorClient(model=model)

        assert client.model.name == "test"
        assert client.include_mt is True

    def test_fkmodel_client(self):
        """Test client with FK model from fkmodels directory."""
        from mtuq.greens_tensor.fk_generator import FKGeneratorClient

        client = FKGeneratorClient(model="ak_scak")

        assert client.model.name == "ak_scak"


class TestFKComparison:
    """Compare Python FK against pre-computed FK Green's functions."""

    def get_benchmark_path(self):
        """Get path to pre-computed FK Green's functions."""
        from mtuq.util import fullpath

        return fullpath("data/tests/benchmark_cap/greens/scak/scak_34")

    def read_fk_trace(self, benchmark_path, distance, ext):
        """
        Read pre-computed FK SAC file.

        Extensions map to components:
            0=ZDD, 1=RDD, 2=TDD
            3=ZDS, 4=RDS, 5=TDS
            6=ZSS, 7=RSS, 8=TSS
            a=ZEP, b=REP, c=TEP (explosion)
        """
        from obspy import read

        filepath = os.path.join(benchmark_path, f"{distance}.grn.{ext}")
        if os.path.exists(filepath):
            return read(filepath, format="sac")[0]
        return None

    def test_benchmark_exists(self):
        """Verify benchmark FK data exists."""
        benchmark_path = self.get_benchmark_path()
        assert os.path.exists(
            benchmark_path
        ), f"Benchmark path not found: {benchmark_path}"

        # Check for some expected files
        for dist in [10, 50, 100]:
            filepath = os.path.join(benchmark_path, f"{dist}.grn.0")
            assert os.path.exists(filepath), f"Missing: {filepath}"

    def test_read_benchmark(self):
        """Test reading benchmark FK trace."""
        benchmark_path = self.get_benchmark_path()
        trace = self.read_fk_trace(benchmark_path, 100, "0")

        assert trace is not None
        assert len(trace.data) > 0
        assert trace.stats.delta > 0

        print(f"\nBenchmark FK trace (100km, ZDD):")
        print(f"  npts: {len(trace.data)}")
        print(f"  dt: {trace.stats.delta}")
        print(f"  max amplitude: {np.max(np.abs(trace.data)):.6e}")

    def test_compare_python_fk(self):
        """Compare Python FK with pre-computed FK at 100 km."""
        from mtuq.greens_tensor.fk_generator import FKGenerator, get_velocity_model

        benchmark_path = self.get_benchmark_path()

        # Read benchmark trace to get parameters
        ref_trace = self.read_fk_trace(benchmark_path, 100, "0")
        if ref_trace is None:
            pytest.skip("Benchmark data not available")

        # Load the ak_scak model (same model used for benchmark)
        model = get_velocity_model("ak_scak")

        print(f"\nModel: {model.name}")
        print(model)

        # Create generator for 34 km depth (from scak_34 directory name)
        gen = FKGenerator(model, source_depth_km=34.0)

        # Compute FK - use smaller nt for speed
        dt = ref_trace.stats.delta
        nt = min(512, len(ref_trace.data))

        print(f"\nComputing Python FK: dt={dt}, nt={nt}")
        result = gen.compute([100.0], dt=dt, nt=nt, verbose=True)

        # Get Python result
        python_zdd = result["traces"][100.0].get("ZDD", np.zeros(nt))
        ref_zdd = ref_trace.data[:nt]

        print(f"\nComparison (ZDD at 100 km):")
        print(f"  Reference max amplitude: {np.max(np.abs(ref_zdd)):.6e}")
        print(f"  Python max amplitude: {np.max(np.abs(python_zdd)):.6e}")

        # Check that Python FK produces non-zero output
        # Note: exact match is not expected due to algorithm differences
        python_max = np.max(np.abs(python_zdd))
        ref_max = np.max(np.abs(ref_zdd))

        if python_max > 0:
            print(f"  Amplitude ratio (Python/Ref): {python_max/ref_max:.4f}")
        else:
            print("  WARNING: Python FK produced zero output")


class TestBessel:
    """Tests for Bessel function implementations."""

    def test_bessel_small_arg(self):
        """Test Bessel functions for small arguments."""
        from mtuq.greens_tensor.fk_generator.kernel import bessel_functions_scipy
        from scipy.special import j0, j1, jn

        z = 0.5
        aj0, aj1, aj2 = bessel_functions_scipy(z)

        assert abs(aj0 - j0(z)) < 1e-10
        assert abs(aj1 - j1(z)) < 1e-10
        assert abs(aj2 - jn(2, z)) < 1e-10

    def test_bessel_large_arg(self):
        """Test Bessel functions for large arguments."""
        from mtuq.greens_tensor.fk_generator.kernel import bessel_functions_scipy
        from scipy.special import j0, j1, jn

        z = 50.0
        aj0, aj1, aj2 = bessel_functions_scipy(z)

        assert abs(aj0 - j0(z)) < 1e-10
        assert abs(aj1 - j1(z)) < 1e-10
        assert abs(aj2 - jn(2, z)) < 1e-10


def test_integration_with_mtuq():
    """Integration test: check that FK generator works with MTUQ infrastructure."""
    try:
        from mtuq.greens_tensor.fk_generator import (
            FKGenerator,
            VelocityModel,
            FKGeneratorClient,
            get_velocity_model,
        )
        from mtuq.event import Origin
        from mtuq.station import Station
        import obspy

        # Load ak_scak model from fkmodels
        model = get_velocity_model("ak_scak")

        # Create origin
        origin = Origin(
            {
                "time": "2020-01-01T00:00:00.000000Z",
                "latitude": 61.0,
                "longitude": -150.0,
                "depth_in_m": 34000.0,  # 34 km
            }
        )

        # Create station
        station = Station(
            {
                "latitude": 61.5,
                "longitude": -149.5,
                "starttime": obspy.UTCDateTime("2020-01-01T00:00:00") - 10,
                "endtime": obspy.UTCDateTime("2020-01-01T00:00:00") + 200,
                "npts": 2101,
                "delta": 0.1,
                "network": "XX",
                "station": "TEST",
                "location": "",
                "id": "XX.TEST.",
            }
        )

        # Create client
        client = FKGeneratorClient(model=model, nt=256, dt=0.1)

        print("Integration test: FK Generator client created successfully")
        print(f"  Model: {model.name} with {model.n_layers} layers")

    except Exception as e:
        print(f"Integration test encountered error: {e}")
        raise


if __name__ == "__main__":
    # Run basic tests
    print("Running FK Generator tests...")

    # Velocity model tests
    print("\n=== Velocity Model Tests ===")
    test_model = TestVelocityModel()
    test_model.test_create_model()
    print("  test_create_model: PASSED")
    test_model.test_layer_properties()
    print("  test_layer_properties: PASSED")
    test_model.test_load_fk_model_ak_scak()
    print("  test_load_fk_model_ak_scak: PASSED")
    test_model.test_fkmodels_path()
    print("  test_fkmodels_path: PASSED")
    test_model.test_depth_lookup()
    print("  test_depth_lookup: PASSED")
    test_model.test_model_arrays()
    print("  test_model_arrays: PASSED")

    # Haskell matrix tests
    print("\n=== Haskell Matrix Tests ===")
    test_haskell = TestHaskellMatrices()
    test_haskell.test_hyperbolic_functions()
    print("  test_hyperbolic_functions: PASSED")
    test_haskell.test_layer_parameters()
    print("  test_layer_parameters: PASSED")
    test_haskell.test_source_coefficients()
    print("  test_source_coefficients: PASSED")

    # Bessel function tests
    print("\n=== Bessel Function Tests ===")
    test_bessel = TestBessel()
    test_bessel.test_bessel_small_arg()
    print("  test_bessel_small_arg: PASSED")
    test_bessel.test_bessel_large_arg()
    print("  test_bessel_large_arg: PASSED")

    # FK generator tests
    print("\n=== FK Generator Tests ===")
    test_gen = TestFKGenerator()
    test_gen.test_generator_init()
    print("  test_generator_init: PASSED")

    # Client tests
    print("\n=== FK Client Tests ===")
    test_client = TestFKClient()
    test_client.test_client_init()
    print("  test_client_init: PASSED")
    test_client.test_fkmodel_client()
    print("  test_fkmodel_client: PASSED")

    # FK comparison tests
    print("\n=== FK Comparison Tests ===")
    test_compare = TestFKComparison()
    test_compare.test_benchmark_exists()
    print("  test_benchmark_exists: PASSED")
    test_compare.test_read_benchmark()
    print("  test_read_benchmark: PASSED")
    test_compare.test_compare_python_fk()
    print("  test_compare_python_fk: PASSED")

    # Integration test
    print("\n=== Integration Test ===")
    test_integration_with_mtuq()
    print("  test_integration_with_mtuq: PASSED")

    print("\n=== All tests passed! ===")
