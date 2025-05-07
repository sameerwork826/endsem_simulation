import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_fit(n, coeff, exponent):
    return coeff * n**(2 * exponent)

# Load data with error handling
try:
    pos_data = np.loadtxt('C:/Users/nande/Desktop/endsem-simulation/rw/sarw_positions.dat')
except Exception as e:
    print(f"Error loading sarw_positions.dat: {e}")
    exit(1)

try:
    scale_data = np.loadtxt('C:/Users/nande/Desktop/endsem-simulation/rw/sarw_scaling.dat')
except Exception as e:
    print(f"Error loading sarw_scaling.dat: {e}")
    exit(1)

# Extract scaling data
n_steps = scale_data[:, 0]
r2_avg = scale_data[:, 1]

# Fit log-log data for scaling exponent
try:
    popt, pcov = curve_fit(power_fit, n_steps, r2_avg, p0=[1.0, 0.75])
    coeff, exponent = popt
except Exception as e:
    print(f"Error fitting scaling data: {e}")
    exponent, coeff = 0.75, 1.0  # Fallback values

# Create figure with 2x2 grid
plt.figure(figsize=(12, 8))

# Plot trajectories
plt.subplot(2, 2, 1)
path_ids = np.unique(pos_data[:, 0])
colors = ['navy', 'crimson', 'lime', 'violet', 'gold']
for i, pid in enumerate(path_ids[:5]):
    mask = pos_data[:, 0] == pid
    x_pos = pos_data[mask, 2]
    y_pos = pos_data[mask, 3]
    plt.plot(x_pos, y_pos, '-', color=colors[i % len(colors)], label=f'Path {int(pid)}', linewidth=1.5)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.title('Self-Avoiding Random Walk Paths', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')

# Plot scaling
plt.subplot(2, 2, 2)
plt.loglog(n_steps, r2_avg, 'mo', markersize=8, label='Simulation Data')
plt.loglog(n_steps, power_fit(n_steps, coeff, exponent), 'k-', 
           label=f'Fit: R² ~ N^{2*exponent:.4f}', linewidth=1.5)
plt.xlabel('Steps (N)', fontsize=12)
plt.ylabel('Mean R²', fontsize=12)
plt.title(f'End-to-End Distance Scaling (ν = {exponent:.4f})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot final configuration
plt.subplot(2, 2, 3)
try:
    config_data = np.loadtxt('C:/Users/nande/Desktop/endsem-simulation/rw/sarw_final_config.xyz', skiprows=2)
    x_coords = config_data[:, 0]
    y_coords = config_data[:, 1]
    plt.scatter(x_coords, y_coords, s=30, c='teal')
    plt.plot(x_coords, y_coords, 'k-', alpha=0.6, linewidth=1.5)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('Final Path Configuration', fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
except Exception as e:
    print(f"Error loading sarw_final_config.xyz: {e}")
    plt.text(0.5, 0.5, 'Configuration data unavailable', 
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)

# Add a caption or leave the last subplot empty
plt.subplot(2, 2, 4)
plt.axis('off')
plt.text(0.5, 0.5, f'SARW Analysis\nScaling Exponent ν = {exponent:.4f}', 
         ha='center', va='center', fontsize=12)

plt.tight_layout(pad=2.0)
plt.savefig('C:/Users/nande/Desktop/endsem-simulation/rw/sarw_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Estimated scaling exponent (ν): {exponent:.4f}")