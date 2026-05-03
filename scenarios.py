"""
Robot Trajectory Optimization - Multiple Scenarios
Demonstrates different start/end points and obstacle configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.patches as patches

def compute_objective(z, T, beta):
    x = z[:T+1]
    y = z[T+1:]
    
    path_length = 0
    for t in range(T):
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        path_length += dx**2 + dy**2
    
    accel_penalty = 0
    for t in range(1, T):
        ddx = x[t+1] - 2*x[t] + x[t-1]
        ddy = y[t+1] - 2*y[t] + y[t-1]
        accel_penalty += ddx**2 + ddy**2
    
    return path_length + beta * accel_penalty

def solve_trajectory(scenario, beta=0.1, T=6):
    """Solve trajectory for a given scenario configuration"""
    x_start = np.array(scenario['start'])
    x_goal = np.array(scenario['goal'])
    obstacles = scenario['obstacles']
    v_max = scenario.get('v_max', 3.0)
    
    t_norm = np.linspace(0, 1, T+1)
    side_amplitude = 3.0
    x0_init = (x_start[0] + t_norm * (x_goal[0] - x_start[0]) + 
               side_amplitude * np.sin(np.pi * t_norm))
    y0_init = (x_start[1] + t_norm * (x_goal[1] - x_start[1]) - 
               side_amplitude * np.sin(np.pi * t_norm))
    z0 = np.concatenate([x0_init, y0_init])
    
    def speed_const(z):
        x = z[:T+1]
        y = z[T+1:]
        speeds = []
        for t in range(T):
            speeds.append((x[t+1]-x[t])**2 + (y[t+1]-y[t])**2)
        return np.array(speeds)
    
    safety_margin = 0.1
    def make_obs_const(obs_center, obs_radius):
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
        return obs_const
    
    A_eq = np.zeros((4, 2*(T+1)))
    A_eq[0, 0] = 1
    A_eq[1, T+1] = 1
    A_eq[2, T] = 1
    A_eq[3, 2*T+1] = 1
    eq_values = np.array([x_start[0], x_start[1], x_goal[0], x_goal[1]])
    
    constraints = [
        LinearConstraint(A_eq, eq_values, eq_values),
        NonlinearConstraint(speed_const, -np.inf, np.full(T, v_max**2))
    ]
    
    n_interp = 4
    n_constraints = (T+1) + T * n_interp
    for obs_center, obs_radius in obstacles:
        constraints.append(NonlinearConstraint(
            make_obs_const(np.array(obs_center), obs_radius),
            np.zeros(n_constraints), np.full(n_constraints, np.inf)
        ))
    
    result = minimize(
        lambda z: compute_objective(z, T, beta),
        z0, method='SLSQP', constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-9}
    )
    
    return result, T

scenarios = {
    "Original": {
        'start': [0, 0],
        'goal': [10, 10],
        'obstacles': [([5, 5], 2.0)],
        'description': 'Centered obstacle (baseline)'
    },
    "Off-center Obstacle": {
        'start': [0, 0],
        'goal': [10, 10],
        'obstacles': [([3, 7], 1.5)],
        'description': 'Asymmetric obstacle position'
    },
    "Two Obstacles": {
        'start': [0, 0],
        'goal': [10, 10],
        'obstacles': [([3, 6], 1.5), ([7, 4], 1.5)],
        'description': 'Robot must navigate around two obstacles'
    },
    "Diagonal Path": {
        'start': [0, 10],
        'goal': [10, 0],
        'obstacles': [([3, 3], 1.5)],
        'description': 'Top-left to bottom-right diagonal'
    },
}

beta = 0.1

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, scenario) in enumerate(scenarios.items()):
    ax = axes[idx]
    result, T = solve_trajectory(scenario, beta=beta)
    
    x_opt = result.x[:T+1]
    y_opt = result.x[T+1:]
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    
    for obs_center, obs_radius in scenario['obstacles']:
        circle = patches.Circle(obs_center, obs_radius, 
                                color='#E74C3C', alpha=0.4, zorder=2)
        ax.add_patch(circle)
        circle_outline = patches.Circle(obs_center, obs_radius, 
                                         fill=False, edgecolor='#C0392B', 
                                         linewidth=2, zorder=3)
        ax.add_patch(circle_outline)
    
    ax.plot(x_opt, y_opt, 'o-', color='#1C7293', linewidth=2.5, 
            markersize=8, zorder=4, label=f'Optimal trajectory')
    
    ax.scatter(scenario['start'][0], scenario['start'][1], 
               color='#02C39A', s=200, zorder=5, marker='o', 
               edgecolors='black', linewidth=2, label='Start')
    ax.scatter(scenario['goal'][0], scenario['goal'][1], 
               color='#FFB800', s=200, zorder=5, marker='*', 
               edgecolors='black', linewidth=2, label='Goal')
    
    path_len = sum(np.sqrt((x_opt[t+1]-x_opt[t])**2 + (y_opt[t+1]-y_opt[t])**2) 
                   for t in range(T))
    
    ax.set_title(f'{name}\n{scenario["description"]}\nPath length: {path_len:.2f} m', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle(f'Trajectory Optimization Across Different Scenarios (β = {beta})', 
             fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('scenarios_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Scenarios comparison saved")

print("\n" + "="*70)
print("SCENARIO RESULTS")
print("="*70)
for name, scenario in scenarios.items():
    result, T = solve_trajectory(scenario, beta=beta)
    x_opt = result.x[:T+1]
    y_opt = result.x[T+1:]
    path_len = sum(np.sqrt((x_opt[t+1]-x_opt[t])**2 + (y_opt[t+1]-y_opt[t])**2) 
                   for t in range(T))
    print(f"\n{name}:")
    print(f"  Description: {scenario['description']}")
    print(f"  Path length: {path_len:.3f} m")
    print(f"  Iterations: {result.nit}")
    print(f"  Converged: {result.success}")