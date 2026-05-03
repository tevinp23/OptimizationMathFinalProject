"""
Convergence Study: Effect of Discretization (T) and Speed Limits (v_max)
Shows robustness of solution across parameter variations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import pandas as pd

# ============================================================================
# CORE OPTIMIZATION FUNCTION
# ============================================================================

def solve_trajectory(T, beta=0.1, v_max=3.0, x_start=np.array([0, 0]), 
                     x_goal=np.array([10, 10]), obs_center=np.array([5, 5]), 
                     obs_radius=2.0):
    """Solve trajectory optimization for given T and v_max"""
    
    # Initial guess
    t_norm = np.linspace(0, 1, T+1)
    side_amplitude = 3.0
    x0_init = (x_start[0] + t_norm * (x_goal[0] - x_start[0]) + 
               side_amplitude * np.sin(np.pi * t_norm))
    y0_init = (x_start[1] + t_norm * (x_goal[1] - x_start[1]) - 
               side_amplitude * np.sin(np.pi * t_norm))
    z0 = np.concatenate([x0_init, y0_init])
    
    # Speed constraint
    def speed_const(z):
        x = z[:T+1]
        y = z[T+1:]
        return np.array([(x[t+1]-x[t])**2 + (y[t+1]-y[t])**2 for t in range(T)])
    
    # Obstacle constraint with interpolation
    safety_margin = 0.1
    effective_radius = obs_radius + safety_margin
    
    def obs_const(z):
        x = z[:T+1]
        y = z[T+1:]
        dists = []
        for t in range(T+1):
            dist_sq = (x[t]-obs_center[0])**2 + (y[t]-obs_center[1])**2
            dists.append(dist_sq - effective_radius**2)
        n_interp = 4
        for t in range(T):
            for i in range(1, n_interp + 1):
                alpha = i / (n_interp + 1)
                px = (1 - alpha) * x[t] + alpha * x[t+1]
                py = (1 - alpha) * y[t] + alpha * y[t+1]
                dist_sq = (px-obs_center[0])**2 + (py-obs_center[1])**2
                dists.append(dist_sq - effective_radius**2)
        return np.array(dists)
    
    # Boundary constraints
    A_eq = np.zeros((4, 2*(T+1)))
    A_eq[0, 0] = 1
    A_eq[1, T+1] = 1
    A_eq[2, T] = 1
    A_eq[3, 2*T+1] = 1
    eq_values = np.array([x_start[0], x_start[1], x_goal[0], x_goal[1]])
    
    n_interp = 4
    n_constraints = (T+1) + T * n_interp
    
    constraints = [
        LinearConstraint(A_eq, eq_values, eq_values),
        NonlinearConstraint(speed_const, -np.inf, np.full(T, v_max**2)),
        NonlinearConstraint(obs_const, np.zeros(n_constraints), np.full(n_constraints, np.inf))
    ]
    
    result = minimize(
        lambda z: compute_objective(z, T, beta),
        z0, method='SLSQP', constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-10, 'maxev': 2000}
    )
    
    return result, T


def compute_objective(z, T, beta):
    x = z[:T+1]
    y = z[T+1:]
    path_length = sum((x[t+1]-x[t])**2 + (y[t+1]-y[t])**2 for t in range(T))
    accel = sum((x[t+1]-2*x[t]+x[t-1])**2 + (y[t+1]-2*y[t]+y[t-1])**2 for t in range(1, T))
    return path_length + beta * accel


def compute_metrics(result, T):
    """Extract all metrics from optimized trajectory"""
    x = result.x[:T+1]
    y = result.x[T+1:]
    
    # Path length
    path_len = sum(np.sqrt((x[t+1]-x[t])**2 + (y[t+1]-y[t])**2) for t in range(T))
    
    # Peak acceleration
    accels = [np.sqrt((x[t+1]-2*x[t]+x[t-1])**2 + (y[t+1]-2*y[t]+y[t-1])**2) 
              for t in range(1, T)]
    peak_accel = max(accels) if accels else 0
    
    # Max speed
    speeds = [np.sqrt((x[t+1]-x[t])**2 + (y[t+1]-y[t])**2) for t in range(T)]
    max_speed = max(speeds)
    
    # Min clearance from obstacle
    obs_center = np.array([5, 5])
    clearances = [np.sqrt((x[t]-obs_center[0])**2 + (y[t]-obs_center[1])**2) - 2.0 
                  for t in range(T+1)]
    min_clearance = min(clearances)
    
    return {
        'path_length': path_len,
        'peak_accel': peak_accel,
        'max_speed': max_speed,
        'min_clearance': min_clearance,
        'objective': result.fun,
        'iterations': result.nit,
        'converged': result.success
    }


# ============================================================================
# STUDY 1: VARYING T (Discretization Convergence)
# ============================================================================

print("="*80)
print("STUDY 1: Convergence with Discretization Level T")
print("="*80)

T_values = [4, 6, 8, 10, 12]
beta = 0.1
results_T = {}

for T in T_values:
    result, _ = solve_trajectory(T, beta=beta)
    metrics = compute_metrics(result, T)
    results_T[T] = metrics
    print(f"\nT = {T} waypoints:")
    print(f"  Path length:    {metrics['path_length']:.4f} m")
    print(f"  Peak accel:     {metrics['peak_accel']:.4f} m/s²")
    print(f"  Max speed:      {metrics['max_speed']:.4f} m/s")
    print(f"  Min clearance:  {metrics['min_clearance']:.4f} m")
    print(f"  Objective:      {metrics['objective']:.4f}")
    print(f"  Iterations:     {metrics['iterations']}")
    print(f"  Converged:      {metrics['converged']}")

# ============================================================================
# STUDY 2: VARYING v_max (Speed Constraints)
# ============================================================================

print("\n" + "="*80)
print("STUDY 2: Robustness to Speed Limit Variations")
print("="*80)

v_max_values = [2.0, 2.2, 2.4, 2.5, 2.7, 3.0, 3.5, 4.0]
T = 6  # Use fixed T from original project
results_vmax = {}

for v_max in v_max_values:
    result, _ = solve_trajectory(T, beta=beta, v_max=v_max)
    metrics = compute_metrics(result, T)
    results_vmax[v_max] = metrics
    print(f"\nv_max = {v_max:.1f} m/s:")
    print(f"  Path length:    {metrics['path_length']:.4f} m")
    print(f"  Peak accel:     {metrics['peak_accel']:.4f} m/s²")
    print(f"  Max speed used: {metrics['max_speed']:.4f} m/s")
    print(f"  Min clearance:  {metrics['min_clearance']:.4f} m")
    print(f"  Objective:      {metrics['objective']:.4f}")
    print(f"  Converged:      {metrics['converged']}")

# ============================================================================
# VISUALIZATION 1: Convergence with T
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

T_list = sorted(results_T.keys())
path_lens = [results_T[T]['path_length'] for T in T_list]
peak_accels = [results_T[T]['peak_accel'] for T in T_list]
max_speeds = [results_T[T]['max_speed'] for T in T_list]
min_clears = [results_T[T]['min_clearance'] for T in T_list]

# Path length convergence
ax = axes[0, 0]
ax.plot(T_list, path_lens, 'o-', linewidth=2.5, markersize=10, color='#1C7293', label='Path length')
ax.axhline(y=path_lens[-1], color='gray', linestyle='--', alpha=0.5, label='Converged value')
ax.set_xlabel('Discretization Level T', fontsize=11, fontweight='bold')
ax.set_ylabel('Path Length (m)', fontsize=11, fontweight='bold')
ax.set_title('Convergence: Path Length', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Peak acceleration convergence
ax = axes[0, 1]
ax.plot(T_list, peak_accels, 'o-', linewidth=2.5, markersize=10, color='#E74C3C', label='Peak acceleration')
ax.axhline(y=peak_accels[-1], color='gray', linestyle='--', alpha=0.5, label='Converged value')
ax.set_xlabel('Discretization Level T', fontsize=11, fontweight='bold')
ax.set_ylabel('Peak Acceleration (m/s²)', fontsize=11, fontweight='bold')
ax.set_title('Convergence: Peak Acceleration', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Max speed convergence
ax = axes[1, 0]
ax.plot(T_list, max_speeds, 'o-', linewidth=2.5, markersize=10, color='#F39C12', label='Max speed achieved')
ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Speed limit (3.0 m/s)')
ax.set_xlabel('Discretization Level T', fontsize=11, fontweight='bold')
ax.set_ylabel('Maximum Speed (m/s)', fontsize=11, fontweight='bold')
ax.set_title('Speed Constraint Satisfaction', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Min clearance convergence
ax = axes[1, 1]
ax.plot(T_list, min_clears, 'o-', linewidth=2.5, markersize=10, color='#27AE60', label='Min clearance')
ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Obstacle boundary')
ax.set_xlabel('Discretization Level T', fontsize=11, fontweight='bold')
ax.set_ylabel('Minimum Clearance (m)', fontsize=11, fontweight='bold')
ax.set_title('Obstacle Avoidance Robustness', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.suptitle(f'Convergence Study: Effect of Discretization Level T (β = {beta})', 
             fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('convergence_T.png', dpi=150, bbox_inches='tight')
print("\n✓ Convergence plot (T) saved")
plt.close()

# ============================================================================
# VISUALIZATION 2: Robustness to Speed Limits
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

vmax_list = sorted(results_vmax.keys())
path_lens_v = [results_vmax[v]['path_length'] for v in vmax_list]
peak_accels_v = [results_vmax[v]['peak_accel'] for v in vmax_list]
max_speeds_v = [results_vmax[v]['max_speed'] for v in vmax_list]
min_clears_v = [results_vmax[v]['min_clearance'] for v in vmax_list]

# Path length vs speed limit
ax = axes[0, 0]
ax.plot(vmax_list, path_lens_v, 'o-', linewidth=2.5, markersize=10, color='#1C7293')
ax.set_xlabel('Speed Limit v_max (m/s)', fontsize=11, fontweight='bold')
ax.set_ylabel('Path Length (m)', fontsize=11, fontweight='bold')
ax.set_title('Path Length vs Speed Constraint', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Peak acceleration vs speed limit
ax = axes[0, 1]
ax.plot(vmax_list, peak_accels_v, 'o-', linewidth=2.5, markersize=10, color='#E74C3C')
ax.set_xlabel('Speed Limit v_max (m/s)', fontsize=11, fontweight='bold')
ax.set_ylabel('Peak Acceleration (m/s²)', fontsize=11, fontweight='bold')
ax.set_title('Peak Acceleration vs Speed Constraint', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Actual max speed vs limit
ax = axes[1, 0]
ax.plot(vmax_list, max_speeds_v, 'o-', linewidth=2.5, markersize=10, color='#F39C12', label='Actual max speed')
ax.plot(vmax_list, vmax_list, '--', linewidth=2, color='red', alpha=0.7, label='Speed limit')
ax.set_xlabel('Speed Limit v_max (m/s)', fontsize=11, fontweight='bold')
ax.set_ylabel('Speed (m/s)', fontsize=11, fontweight='bold')
ax.set_title('Speed Constraint Utilization', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Min clearance vs speed limit
ax = axes[1, 1]
ax.plot(vmax_list, min_clears_v, 'o-', linewidth=2.5, markersize=10, color='#27AE60')
ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Speed Limit v_max (m/s)', fontsize=11, fontweight='bold')
ax.set_ylabel('Minimum Clearance (m)', fontsize=11, fontweight='bold')
ax.set_title('Obstacle Avoidance vs Speed', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Robustness Study: Effect of Speed Limit v_max (T = {T}, β = {beta})', 
             fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('robustness_vmax.png', dpi=150, bbox_inches='tight')
print("✓ Robustness plot (v_max) saved")
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE: Convergence with T")
print("="*80)

df_T = pd.DataFrame(results_T).T
print(df_T.to_string())

print("\n" + "="*80)
print("SUMMARY TABLE: Robustness to v_max")
print("="*80)

df_vmax = pd.DataFrame(results_vmax).T
print(df_vmax.to_string())

print("\n✓ All studies complete")