from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.patches as patches

# ============================================================================
# PROBLEM SETUP
# ============================================================================

T = 6
x_start = np.array([0.0, 0.0])
x_goal = np.array([10.0, 10.0])
obstacle_center = np.array([5.0, 5.0])
obstacle_radius = 2.0
v_max = 3.0
beta_values = [0.01, 0.1, 1.0]

results = []

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def compute_objective(z, T, beta):
    """
    Objective function:
    J(z) = ||path_length||^2 + beta * ||acceleration||^2
    
    where z = [x_0, ..., x_T, y_0, ..., y_T]
    """
    x = z[:T+1]
    y = z[T+1:]
    
    # Path length term: sum of squared step distances
    path_length = 0
    for t in range(T):
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        path_length += dx**2 + dy**2
    
    # Acceleration term: sum of squared second differences
    accel_penalty = 0
    for t in range(1, T):
        ddx = x[t+1] - 2*x[t] + x[t-1]
        ddy = y[t+1] - 2*y[t] + y[t-1]
        accel_penalty += ddx**2 + ddy**2
    
    return path_length + beta * accel_penalty

# ============================================================================
# CONSTRAINT FUNCTIONS
# ============================================================================

def build_constraints(T, v_max, obs_center, obs_radius):
    """
    Construct inequality constraints:
    1. Speed limit: ||v_t||^2 <= v_max^2
    2. Obstacle avoidance: ||x_t - c||^2 >= r^2
    """
    
    def speed_constraints(z):
        """Returns ||v_t||^2 for each t (must be <= v_max^2)"""
        x = z[:T+1]
        y = z[T+1:]
        constraints = np.zeros(T)
        for t in range(T):
            dx = x[t+1] - x[t]
            dy = y[t+1] - y[t]
            constraints[t] = dx**2 + dy**2
        return constraints
    
    def obstacle_constraints(z):
        """Returns ||x_t - c||^2 for each t (must be >= r^2)"""
        x = z[:T+1]
        y = z[T+1:]
        constraints = np.zeros(T+1)
        cx, cy = obs_center
        for t in range(T+1):
            dist_sq = (x[t] - cx)**2 + (y[t] - cy)**2
            constraints[t] = dist_sq
        return constraints
    
    return speed_constraints, obstacle_constraints

# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

print("=" * 75)
print(" " * 15 + "ROBOT TRAJECTORY OPTIMIZATION")
print("=" * 75)
print(f"\nProblem Configuration:")
print(f"  Time horizon T = {T}")
print(f"  Start: {x_start}, Goal: {x_goal}")
print(f"  Obstacle: center = {obstacle_center}, radius = {obstacle_radius}")
print(f"  Speed limit v_max = {v_max}")

for b_idx, beta in enumerate(beta_values):
    print(f"\n{'='*75}")
    print(f"Case {b_idx+1}: β = {beta}")
    print(f"{'='*75}")
    
    # ---- Initial Guess ----
    t_norm = np.linspace(0, 1, T+1)
    
    # Sinusoidal detour away from direct line
    side_amplitude = 3.0
    x0_init = (x_start[0] + t_norm * (x_goal[0] - x_start[0]) + 
               side_amplitude * np.sin(np.pi * t_norm))
    y0_init = (x_start[1] + t_norm * (x_goal[1] - x_start[1]) - 
               side_amplitude * np.sin(np.pi * t_norm))
    
    z0 = np.concatenate([x0_init, y0_init])
    
    # ---- Objective Function ----
    obj = lambda z: compute_objective(z, T, beta)
    
    # ---- Constraint Functions ----
    speed_const, obs_const = build_constraints(T, v_max, obstacle_center, obstacle_radius)
    
    # Speed constraint: g(z) = speed^2 <= v_max^2
    speed_ub = np.full(T, v_max**2)
    speed_constraint = NonlinearConstraint(speed_const, -np.inf, speed_ub)
    
    # Obstacle constraint: g(z) = dist^2 >= r^2
    obs_lb = np.full(T+1, obstacle_radius**2)
    obs_constraint = NonlinearConstraint(obs_const, obs_lb, np.inf)
    
    # Boundary conditions (equality)
    eq_indices = [0, T+1, T, 2*T+1]
    eq_values = [x_start[0], x_start[1], x_goal[0], x_goal[1]]
    
    A_eq = np.zeros((4, 2*T+2))
    for i, idx in enumerate(eq_indices):
        A_eq[i, idx] = 1.0
    eq_constraint = LinearConstraint(A_eq, eq_values, eq_values)
    
    # ---- Solve ----
    result = minimize(
        obj, z0,
        method='SLSQP',
        constraints=[eq_constraint, speed_constraint, obs_constraint],
        options={'ftol': 1e-7, 'maxiter': 800}
    )
    
    z_opt = result.x
    J_opt = result.fun
    x_opt = z_opt[:T+1]
    y_opt = z_opt[T+1:]
    
    # ---- Compute Metrics ----
    path_length = 0.0
    for t in range(T):
        path_length += np.sqrt((x_opt[t+1]-x_opt[t])**2 + (y_opt[t+1]-y_opt[t])**2)
    
    accel_penalty = 0.0
    for t in range(1, T):
        ddx = x_opt[t+1] - 2*x_opt[t] + x_opt[t-1]
        ddy = y_opt[t+1] - 2*y_opt[t] + y_opt[t-1]
        accel_penalty += np.sqrt(ddx**2 + ddy**2)
    
    max_speed = 0.0
    speed_array = []
    for t in range(T):
        speed = np.sqrt((x_opt[t+1]-x_opt[t])**2 + (y_opt[t+1]-y_opt[t])**2)
        speed_array.append(speed)
        max_speed = max(max_speed, speed)
    
    print(f"\nOptimization Results:")
    print(f"  Objective value (J):     {J_opt:.6f}")
    print(f"  Path length:             {path_length:.6f}")
    print(f"  Acceleration penalty:    {accel_penalty:.6f}")
    print(f"  Max speed:               {max_speed:.6f} (limit: {v_max})")
    print(f"  Convergence:             {'✓ Success' if result.success else '✗ Did not converge'}")
    print(f"  Iterations:              {result.nit}")
    
    results.append({
        'beta': beta,
        'x': x_opt,
        'y': y_opt,
        'J': J_opt,
        'path_length': path_length,
        'accel_penalty': accel_penalty,
        'max_speed': max_speed,
        'speed_array': speed_array,
        'success': result.success
    })

# ============================================================================
# VISUALIZATION 1: Trajectories for Different β Values
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('Effect of Smoothness Parameter β on Trajectory', 
             fontsize=16, fontweight='bold', y=1.02)

for b_idx, res in enumerate(results):
    ax = axes[b_idx]
    
    x, y = res['x'], res['y']
    
    # Trajectory
    ax.plot(x, y, 'b-o', linewidth=2.5, markersize=6, label='Trajectory', zorder=3)
    
    # Start and goal
    ax.plot(x_start[0], x_start[1], 'gs', markersize=14, linewidth=2.5, 
            label='Start', zorder=4, markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax.plot(x_goal[0], x_goal[1], 'r*', markersize=20, linewidth=2.5, 
            label='Goal', zorder=4)
    
    # Obstacle
    circle = patches.Circle(obstacle_center, obstacle_radius, fill=True, 
                           facecolor='red', alpha=0.25, edgecolor='darkred', 
                           linewidth=2.5, label='Obstacle', zorder=2)
    ax.add_patch(circle)
    
    # Formatting
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'β = {res["beta"]:.3f}\nJ = {res["J"]:.3f}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-1.5, 11.5)
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('trajectory_beta_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: trajectory_beta_comparison.png")

# ============================================================================
# VISUALIZATION 2: Metrics Analysis
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('Performance Metrics vs Smoothness Parameter β', 
             fontsize=16, fontweight='bold', y=1.02)

betas = np.array([r['beta'] for r in results])
path_lengths = np.array([r['path_length'] for r in results])
accel_penalties = np.array([r['accel_penalty'] for r in results])
max_speeds = np.array([r['max_speed'] for r in results])

# Path length vs beta
axes[0].loglog(betas, path_lengths, 'bo-', linewidth=3, markersize=10, markerfacecolor='lightblue', markeredgewidth=2)
axes[0].set_xlabel('β (log scale)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Path Length (m)', fontsize=11, fontweight='bold')
axes[0].set_title('Path Length vs β', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.4, which='both')
axes[0].tick_params(labelsize=10)

# Acceleration penalty vs beta
axes[1].loglog(betas, accel_penalties, 'ro-', linewidth=3, markersize=10, markerfacecolor='lightcoral', markeredgewidth=2)
axes[1].set_xlabel('β (log scale)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Acceleration Penalty', fontsize=11, fontweight='bold')
axes[1].set_title('Acceleration Penalty vs β', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.4, which='both')
axes[1].tick_params(labelsize=10)

# Max speed vs beta
axes[2].semilogx(betas, max_speeds, 'go-', linewidth=3, markersize=10, markerfacecolor='lightgreen', markeredgewidth=2, label='Max speed')
axes[2].axhline(v_max, color='darkred', linestyle='--', linewidth=2.5, label=f'Speed limit ({v_max})')
axes[2].set_xlabel('β (log scale)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Max Speed (m/s)', fontsize=11, fontweight='bold')
axes[2].set_title('Max Speed vs β', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10, framealpha=0.95)
axes[2].grid(True, alpha=0.4, which='both')
axes[2].set_ylim([0, v_max * 1.4])
axes[2].tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('metrics_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: metrics_analysis.png")

# ============================================================================
# VISUALIZATION 3: Speed Profile Over Time
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('Speed Profile Over Time for Different β Values', 
             fontsize=16, fontweight='bold', y=1.02)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for b_idx, res in enumerate(results):
    ax = axes[b_idx]
    times = np.arange(len(res['speed_array']))
    
    ax.plot(times, res['speed_array'], 'o-', color=colors[b_idx], linewidth=2.5, 
            markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax.axhline(v_max, color='red', linestyle='--', linewidth=2, label='Speed limit')
    ax.fill_between(times, 0, v_max, alpha=0.1, color='green', label='Feasible region')
    
    ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speed (m/s)', fontsize=11, fontweight='bold')
    ax.set_title(f'β = {res["beta"]:.3f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.set_ylim([0, v_max * 1.3])
    ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('speed_profiles.png', dpi=150, bbox_inches='tight')
print("✓ Saved: speed_profiles.png")

# ============================================================================
# VISUALIZATION 4: Trade-off Analysis
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

path_vs_accel = [
    (r['path_length'], r['accel_penalty'], r['beta']) 
    for r in results
]

betas_plot = [x[2] for x in path_vs_accel]
paths = [x[0] for x in path_vs_accel]
accels = [x[1] for x in path_vs_accel]

scatter = ax.scatter(paths, accels, s=300, c=np.log10(betas_plot), cmap='viridis',
                    alpha=0.8, edgecolors='black', linewidths=2, zorder=3)

# Annotate points
for i, (p, a, b) in enumerate(path_vs_accel):
    ax.annotate(f'β={b:.3f}', (p, a), xytext=(8, 8), textcoords='offset points',
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Path Length (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Acceleration Penalty', fontsize=12, fontweight='bold')
ax.set_title('Pareto Trade-off: Path Length vs Smoothness', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=11)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('log₁₀(β)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('tradeoff_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: tradeoff_analysis.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print(f"\n{'='*75}")
print(" " * 20 + "SUMMARY OF RESULTS")
print(f"{'='*75}\n")

print("Effect of β (Smoothness Parameter):")
print(f"{'β':<10} {'Objective':<13} {'Path Length':<14} {'Accel Penalty':<15} {'Max Speed':<12}")
print("-" * 64)
for r in results:
    print(f"{r['beta']:<10.3f} {r['J']:<13.4f} {r['path_length']:<14.4f} {r['accel_penalty']:<15.4f} {r['max_speed']:<12.4f}")

print(f"\n{'='*75}\n")