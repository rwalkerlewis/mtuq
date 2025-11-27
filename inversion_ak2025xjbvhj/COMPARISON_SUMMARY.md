# Moment Tensor Inversion Results Comparison

## Event: ak2025xjbvhj (Alaska M6.0 Deep Earthquake)
- **Origin Time:** 2025-11-27 17:11:29 UTC
- **Location:** 61.57°N, 150.75°W
- **Depth:** 69.4 km

## Inversion Configurations Tested

| Run | Stations | Misfit | VR | Mw | Stability | Notes |
|-----|----------|--------|-----|-----|-----------|-------|
| Original | 50 | ~0.96 | ~4% | 6.10 | - | All available stations |
| Filtered | 34 | ~0.96 | ~4% | 6.10 | - | Removed worst-fitting stations |
| ⚠️ Overfit | 10 | 0.173 | 83% | 6.00 | - | **OVERFIT** - VR-based selection |
| ✓ **Robust** | **30** | **0.548** | **45%** | **6.10** | **±0.00** | SNR-based, jackknife-validated |

## Understanding Overfitting vs. Robust Solutions

### ⚠️ Overfit Solution (10 stations, VR=83%)
The "optimized" solution with only 10 stations achieved 82.7% VR but represents **overfitting**:
- Stations were selected based on **how well they fit a particular solution**
- This creates circular logic: we select stations that agree with our answer
- The solution may not generalize to independent data
- Low misfit doesn't mean the solution is physically correct

### ✓ Robust Solution (30 stations, VR=45%)
The robust approach provides a more reliable result:
- Stations selected based on **signal-to-noise ratio (SNR)** - independent of any MT solution
- Uses more stations (30 vs 10) for better azimuthal coverage
- **Jackknife stability analysis**: Mw = 6.10 ± 0.00 when dropping individual stations
- 45% VR is typical for regional MT inversions with 1D Green's functions

## Robust Inversion Methodology

1. **SNR-based station selection**: Stations with SNR ≥ 50 selected (solution-independent)
2. **Azimuthal coverage**: Ensured stations span different azimuths
3. **Jackknife stability analysis**: Tested solution stability by dropping one station at a time
4. **Feature-based quality metrics**: Checked cross-correlation and amplitude ratios

## Final Robust Solution

| Parameter | Value |
|-----------|-------|
| **Magnitude (Mw)** | 6.10 |
| **Seismic Moment** | 1.78×10¹⁸ N·m |
| **Strike** | 75° |
| **Dip** | 57° |
| **Slip (Rake)** | 7.5° |
| **Misfit** | 0.548 |
| **Variance Reduction** | 45.2% |
| **Stability (±Mw)** | ±0.00 |

### Moment Tensor Components (N·m)
- Mrr: -7.16×10¹⁶
- Mtt: -5.77×10¹⁷
- Mpp: +1.14×10¹⁸
- Mrt: +3.39×10¹⁷
- Mrp: +1.13×10¹⁸
- Mtp: +9.74×10¹⁷

## Key Insights

1. **Lower misfit ≠ better solution**: The overfit solution had lower misfit but used circular station selection
2. **Stability matters**: The robust solution is completely stable under jackknife resampling
3. **More stations = more robust**: Using 30 stations provides better azimuthal coverage and reduces bias
4. **VR ~45% is realistic**: For regional MT inversions with 1D Green's functions, this is a typical value

## Recommendations

For reliable moment tensor inversions:
- Select stations based on data quality (SNR), not fit to solution
- Use as many high-quality stations as possible
- Validate solution stability with jackknife or bootstrap resampling
- Don't chase low misfit at the expense of robustness
