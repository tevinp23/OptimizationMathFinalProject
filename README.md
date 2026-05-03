# Robot Trajectory Optimization via Constrained Nonlinear Programming

## Overview

This project solves a robot trajectory optimization problem using constrained nonlinear programming. The objective balances path-length minimization with an acceleration penalty to achieve smooth, efficient trajectories while respecting speed limits and avoiding obstacles.

## Problem Description

A planar robot must navigate from (0, 0) to (10, 10) in a 10×10 meter environment while:
- Minimizing total path length
- Minimizing acceleration (trajectory smoothness)
- Respecting a speed limit of 3.0 m/s
- Avoiding circular obstacles
- Discretizing the trajectory into 7 waypoints (T=6 time steps)

The solution uses Sequential Least Squares Programming (SLSQP) and verifies optimality via Karush-Kuhn-Tucker (KKT) conditions.

## Key Results

- **Efficiency-Smoothness Trade-off**: Increasing the smoothness parameter β from 0.01 to 1.0 reduces peak acceleration by 36% while adding only 0.24% to path length
- **Optimal Discretization**: Seven waypoints (T=6) provide sufficient solution fidelity with negligible computational cost
- **Binding Constraints**: The obstacle geometry dominates the solution; the speed limit is slack. The robot needs a minimum speed of 2.5 m/s to navigate safely but uses only 2.48 m/s regardless of higher allowances
- **Generalization**: The solver successfully handles diverse problem geometries including off-center obstacles, multiple obstacles, and diagonal paths

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pandas

### Setup

```bash
git clone https://github.com/tevinp23/OptimizationMathFinalProject/
cd trajectory-optimization
pip install numpy scipy matplotlib pandas
```

## Usage

Run all three programs to generate all figures:

```bash
python3 trajectory_optimizer.py
python3 convergence_study.py
python3 scenarios.py
```

### Program Descriptions

#### 1. `trajectory_optimizer.py`
Solves the baseline trajectory optimization for three smoothness parameters (β = 0.01, 0.1, 1.0).

**Output:**
- `trajectory_beta_comparison.png` - Trajectories for each β value
- `metrics_analysis.png` - Metrics vs β
- Console output with detailed results

#### 2. `convergence_study.py`
Tests solution sensitivity to discretization level (T) and speed limits (v_max).

**Output:**
- `convergence_T.png` - Convergence analysis
- `robustness_vmax.png` - Speed robustness analysis
- Summary tables

#### 3. `scenarios.py`
Tests the solver on four different problem configurations:
- Baseline (centered obstacle)
- Off-center obstacle
- Two obstacles
- Diagonal path

**Output:**
- `scenarios_comparison.png` - All four scenarios

## File Structure

```
trajectory-optimization/
├── trajectory_optimizer.py
├── convergence_study.py
├── scenarios.py
├── trajectory_beta_comparison.png
├── metrics_analysis.png
├── convergence_T.png
├── scenarios_comparison.png
├── robustness_vmax.png
├── speed_profiles.png
├── tradeoff_analysis.png
├── README.md
└── Robot_Trajectory_Optimization_Report.docx
```

## Generated Outputs

### Main Figures (in Paper)

1. **trajectory_beta_comparison.png** - Effect of β on trajectory
2. **metrics_analysis.png** - Path, acceleration, speed vs β
3. **convergence_T.png** - Convergence with discretization
4. **scenarios_comparison.png** - Four problem scenarios
5. **robustness_vmax.png** - Robustness to speed limits

### Extra Figures

- **speed_profiles.png** - Speed over time
- **tradeoff_analysis.png** - Path vs acceleration trade-off

## Mathematical Setup

**Decision Variables:** z = [x₀, ..., x_T, y₀, ..., y_T] ∈ ℝ¹⁴

**Objective:**
```
J(z) = Σ ||x_{t+1} - x_t||²_2 + β Σ ||x_{t+1} - 2x_t + x_{t-1}||²
```

**Constraints:**
- Boundary: x₀ = (0,0), x_T = (10,10)
- Speed: ||v_t||₂ ≤ 3 m/s
- Obstacle: ||x_t - c||₂ ≥ r

**Solver:** SLSQP with BFGS Hessian approximation

## Example Output

```
================================================================================
ROBOT TRAJECTORY OPTIMIZATION
================================================================================

Case 1: β = 0.01
Path length: 14.697 m
Peak acceleration: 1.316 m/s²
Max speed: 2.451 m/s
Converged: True

Case 2: β = 0.1
Path length: 14.700 m
Peak acceleration: 1.194 m/s²
Max speed: 2.460 m/s
Converged: True

Case 3: β = 1.0
Path length: 14.734 m
Peak acceleration: 0.842 m/s²
Max speed: 2.503 m/s
Converged: True
```

## For More Details

See **Robot_Trajectory_Optimization_Report.docx** for the full paper with mathematical formulation, results, and discussion.


