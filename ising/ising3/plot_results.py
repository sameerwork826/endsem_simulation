import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read and plot results
def plot_results():
    # File names for the three magnetic field values
    h_values = [0.0, 0.005, 0.01]
    file_names = [f"ising_L40_h{h:.6f}_ensemble.csv" for h in h_values]
    
    # Create a figure with two subplots (magnetization and susceptibility)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors for different h_z values
    colors = ['orange', 'purple', 'black']
    
    # Plot both quantities for each h_z value
    for i, (h, filename, color) in enumerate(zip(h_values, file_names, colors)):
        try:
            # Read data
            data = pd.read_csv(filename)
            
            # Extract columns
            temp = data['Temperature']
            mag = data['AbsoluteMagnetization']
            mag_err = data['MagnetizationError']
            sus = data['Susceptibility']
            sus_err = data['SusceptibilityError']
            
            # Plot magnetization with error bars
            ax1.errorbar(temp, mag, yerr=mag_err, fmt='o-', color=color, 
                         label=f'$h_z = {h}$', capsize=3, markersize=4)
            
            # Plot susceptibility with error bars
            ax2.errorbar(temp, sus, yerr=sus_err, fmt='o-', color=color, 
                         label=f'$h_z = {h}$', capsize=3, markersize=4)
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping this dataset.")
    
    # Set titles and labels for magnetization plot
    ax1.set_title('Absolute Magnetization vs Temperature')
    ax1.set_xlabel('Temperature ($T$)')
    ax1.set_ylabel('Absolute Magnetization per Spin ($|M|$)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Set titles and labels for susceptibility plot
    ax2.set_title('Magnetic Susceptibility vs Temperature')
    ax2.set_xlabel('Temperature ($T$)')
    ax2.set_ylabel('Susceptibility ($\\chi$)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig('ising_model_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_results()