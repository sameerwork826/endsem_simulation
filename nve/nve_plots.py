import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def straight_line_fit(t, m, c):
    return m * t + c

# Load simulation data
sim_energy = np.loadtxt(r'C:\Users\nande\Desktop\endsem-simulation\nve\lj_sim_energy.dat')
sim_msd = np.loadtxt(r'C:\Users\nande\Desktop\endsem-simulation\nve\lj_sim_msd.dat')

# Extract data columns
sim_time = sim_energy[:, 0]
pot_energy = sim_energy[:, 1]
kin_energy = sim_energy[:, 2]
tot_energy = sim_energy[:, 3]
temp = sim_energy[:, 4]

msd_t = sim_msd[:, 0]
msd_values = sim_msd[:, 1]

# Create figure with adjusted layout
plt.figure(figsize=(12, 10))

# Plot energy components
plt.subplot(2, 2, 1)
plt.plot(sim_time, pot_energy, 'b-', label='Potential Energy')
plt.plot(sim_time, kin_energy, 'orange', label='Kinetic Energy')
plt.plot(sim_time, tot_energy, 'purple', label='Total Energy')
plt.xlabel('Time (ps)')
plt.ylabel('Energy (eV/particle)')
plt.title('Energy Components Evolution')
plt.legend()
plt.grid(True)

# Plot temperature
plt.subplot(2, 2, 2)
plt.plot(sim_time, temp, 'm-')
plt.xlabel('Time (ps)')
plt.ylabel('Temperature (K)')
plt.title('System Temperature Evolution')
plt.grid(True)

# Plot MSD on log scale
plt.subplot(2, 2, 3)
# Filter out zero values for log plot
valid_idx = msd_values > 0
log_t = np.log(msd_t[valid_idx])
log_msd_val = np.log(msd_values[valid_idx])

# Fit log(MSD) vs log(t) to get scaling
fit_start = max(10, int(len(log_t) * 0.1))
params, cov = curve_fit(straight_line_fit, log_t[fit_start:], log_msd_val[fit_start:])
exponent = params[0]
offset = params[1]

# Plot log-log with fit
plt.loglog(msd_t, msd_values, 'go', markersize=4, label='MSD Data')
plt.loglog(msd_t, np.exp(offset) * msd_t**exponent, 'k-', 
           label=f'Fit: MSD ~ t^{exponent:.4f}')
plt.xlabel('Time (ps, log scale)')
plt.ylabel('MSD (Å², log scale)')
plt.title(f'Mean Squared Displacement\nExponent = {exponent:.4f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lj_simulation_analysis.png', dpi=300)
plt.show()

print(f"MSD scaling exponent (α): {exponent:.4f}")
print(f"Mean temperature: {np.mean(temp):.4f} K")

# Plot final particle configuration
try:
    config = np.loadtxt(r'C:\Users\nande\Desktop\endsem-simulation\nve\lj_sim_final_config.dat', skiprows=2)
    x_pos = config[:, 1]
    y_pos = config[:, 2]
    
    plt.figure(figsize=(7, 7))
    plt.scatter(x_pos, y_pos, s=15, c='red')
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.xlabel('X coordinate (Å)')
    plt.ylabel('Y coordinate (Å)')
    plt.title('Final Particle Positions')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('lj_final_positions.png', dpi=300)
    plt.show()
except:
    print("Failed to load configuration file.")