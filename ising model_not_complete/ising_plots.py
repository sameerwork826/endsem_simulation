import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)

# Read data from file
def read_data(filename="ising_data.txt"):
    file_path = os.path.join(SCRIPT_DIR, filename)
    data = pd.read_csv(file_path, sep=" ", comment="#", 
                      names=["Ensemble", "Temperature", "Energy", "Magnetization"])
    return data

# Calculate ensemble averages with error bars
def calculate_ensemble_averages(data):
    grouped = data.groupby("Ensemble")
    ensemble_results = []
    
    for name, group in grouped:
        group['Mag_Abs'] = group['Magnetization'].abs()
        group['Mag_Squared'] = group['Magnetization'] ** 2
        group['Energy_Squared'] = group['Energy'] ** 2
        
        temperature = group['Temperature'].iloc[0]
        
        avg_energy = group['Energy'].mean()
        err_energy = group['Energy'].std() / np.sqrt(len(group))
        
        avg_mag = group['Magnetization'].mean()
        err_mag = group['Magnetization'].std() / np.sqrt(len(group))
        
        avg_abs_mag = group['Mag_Abs'].mean()
        err_abs_mag = group['Mag_Abs'].std() / np.sqrt(len(group))
        
        avg_mag_squared = group['Mag_Squared'].mean()
        err_mag_squared = group['Mag_Squared'].std() / np.sqrt(len(group))
        
        specific_heat = (group['Energy_Squared'].mean() - avg_energy**2) / (temperature**2)
        
        susceptibility = (avg_mag_squared - avg_mag**2) / temperature
        
        ensemble_results.append({
            'Ensemble': name,
            'Temperature': temperature,
            'Energy': avg_energy,
            'Energy_Error': err_energy,
            'Magnetization': avg_mag,
            'Magnetization_Error': err_mag,
            'Abs_Magnetization': avg_abs_mag,
            'Abs_Magnetization_Error': err_abs_mag,
            'Specific_Heat': specific_heat,
            'Susceptibility': susceptibility
        })
    
    return pd.DataFrame(ensemble_results)

# Calculate overall averages across all ensembles
def calculate_overall_averages(ensemble_results):
    grouped = ensemble_results.groupby('Temperature')
    overall_results = []
    
    for temp, group in grouped:
        result = {
            'Temperature': temp,
            'Energy': group['Energy'].mean(),
            'Energy_Error': np.sqrt((group['Energy_Error']**2).mean() + group['Energy'].std()**2),
            'Magnetization': group['Magnetization'].mean(),
            'Magnetization_Error': np.sqrt((group['Magnetization_Error']**2).mean() + group['Magnetization'].std()**2),
            'Abs_Magnetization': group['Abs_Magnetization'].mean(),
            'Abs_Magnetization_Error': np.sqrt((group['Abs_Magnetization_Error']**2).mean() + group['Abs_Magnetization'].std()**2),
            'Specific_Heat': group['Specific_Heat'].mean(),
            'Specific_Heat_Error': group['Specific_Heat'].std(),
            'Susceptibility': group['Susceptibility'].mean(),
            'Susceptibility_Error': group['Susceptibility'].std()
        }
        overall_results.append(result)
    
    return pd.DataFrame(overall_results)

# Plot time series for a single ensemble
def plot_time_series(data, ensemble=0):
    ensemble_data = data[data['Ensemble'] == ensemble]
    time_steps = np.arange(len(ensemble_data))
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(time_steps, ensemble_data['Energy'], label='Energy per site')
    axs[0].set_ylabel('Energy per site')
    axs[0].legend()
    
    axs[1].plot(time_steps, ensemble_data['Magnetization'], label='Magnetization per site')
    axs[1].set_xlabel('Monte Carlo Steps')
    axs[1].set_ylabel('Magnetization per site')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'ising_time_series.png'), dpi=300)
    return fig

# Plot spin configuration
def plot_configuration(filename, output_name=None):
    file_path = os.path.join(SCRIPT_DIR, filename)
    config = np.loadtxt(file_path)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(config, cmap='binary', interpolation='nearest')
    plt.colorbar(label='Spin')
    plt.title('Spin Configuration')
    plt.tight_layout()
    
    if output_name:
        plt.savefig(os.path.join(SCRIPT_DIR, output_name), dpi=300)
    else:
        plt.savefig(os.path.join(SCRIPT_DIR, 'ising_configuration.png'), dpi=300)
    return fig

# Plot ensemble averages with error bars
def plot_ensemble_results(ensemble_results):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    for _, row in ensemble_results.iterrows():
        axs[0, 0].errorbar(row['Ensemble'], row['Energy'], yerr=row['Energy_Error'], 
                          fmt='o', capsize=5)
    axs[0, 0].set_xlabel('Ensemble')
    axs[0, 0].set_ylabel('Energy per site')
    axs[0, 0].set_title('Energy per Ensemble')
    
    for _, row in ensemble_results.iterrows():
        axs[0, 1].errorbar(row['Ensemble'], row['Abs_Magnetization'], 
                          yerr=row['Abs_Magnetization_Error'], fmt='o', capsize=5)
    axs[0, 1].set_xlabel('Ensemble')
    axs[0, 1].set_ylabel('|Magnetization| per site')
    axs[0, 1].set_title('Absolute Magnetization per Ensemble')
    
    axs[1, 0].bar(ensemble_results['Ensemble'], ensemble_results['Specific_Heat'])
    axs[1, 0].set_xlabel('Ensemble')
    axs[1, 0].set_ylabel('Specific Heat')
    axs[1, 0].set_title('Specific Heat per Ensemble')
    
    axs[1, 1].bar(ensemble_results['Ensemble'], ensemble_results['Susceptibility'])
    axs[1, 1].set_xlabel('Ensemble')
    axs[1, 1].set_ylabel('Susceptibility')
    axs[1, 1].set_title('Susceptibility per Ensemble')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'ising_ensemble_results.png'), dpi=300)
    return fig

# Plot overall averages
def plot_overall_results(overall_results):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    for _, row in overall_results.iterrows():
        axs[0, 0].errorbar(row['Temperature'], row['Energy'], yerr=row['Energy_Error'], 
                         fmt='o', capsize=5, markersize=10)
    axs[0, 0].set_xlabel('Temperature')
    axs[0, 0].set_ylabel('Energy per site')
    axs[0, 0].set_title('Average Energy')
    
    for _, row in overall_results.iterrows():
        axs[0, 1].errorbar(row['Temperature'], row['Abs_Magnetization'], 
                         yerr=row['Abs_Magnetization_Error'], fmt='o', capsize=5, markersize=10)
    axs[0, 1].set_xlabel('Temperature')
    axs[0, 1].set_ylabel('|Magnetization| per site')
    axs[0, 1].set_title('Average Absolute Magnetization')
    
    for _, row in overall_results.iterrows():
        axs[1, 0].errorbar(row['Temperature'], row['Specific_Heat'], 
                         yerr=row['Specific_Heat_Error'], fmt='o', capsize=5, markersize=10)
    axs[1, 0].set_xlabel('Temperature')
    axs[1, 0].set_ylabel('Specific Heat')
    axs[1, 0].set_title('Average Specific Heat')
    
    for _, row in overall_results.iterrows():
        axs[1, 1].errorbar(row['Temperature'], row['Susceptibility'], 
                         yerr=row['Susceptibility_Error'], fmt='o', capsize=5, markersize=10)
    axs[1, 1].set_xlabel('Temperature')
    axs[1, 1].set_ylabel('Susceptibility')
    axs[1, 1].set_title('Average Susceptibility')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'ising_overall_results.png'), dpi=300)
    return fig

# Create a PDF report with all results
def create_pdf_report(ensemble_results, overall_results, data):
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(os.path.join(SCRIPT_DIR, 'ising_model_report.pdf')) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.5, 'Ising Model Simulation Report\n\n'
                'PHY-407: Simulation Methods in Statistical Physics\n'
                'IDD Part-IV (Session: 2024-25)', 
                ha='center', va='center', fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Plot time series
        fig_time_series = plot_time_series(data)
        pdf.savefig(fig_time_series)
        plt.close(fig_time_series)
        
        # Plot ensemble results
        fig_ensemble = plot_ensemble_results(ensemble_results)
        pdf.savefig(fig_ensemble)
        plt.close(fig_ensemble)
        
        # Plot overall results
        fig_overall = plot_overall_results(overall_results)
        pdf.savefig(fig_overall)
        plt.close(fig_overall)
        
        # Configuration plots
        for i in range(len(ensemble_results)):
            fig_config = plot_configuration(f'config_ensemble_{i}.txt', f'config_{i}.png')
            pdf.savefig(fig_config)
            plt.close(fig_config)
        
        # Table of results
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Summary of Results', ha='center', fontsize=14)
        
        table_data = []
        headers = ['Temperature', 'Energy', 'Magnetization', 'Specific Heat', 'Susceptibility']
        table_data.append(headers)
        
        for _, row in overall_results.iterrows():
            table_data.append([
                f"{row['Temperature']:.2f}",
                f"{row['Energy']:.4f} ± {row['Energy_Error']:.4f}",
                f"{row['Abs_Magnetization']:.4f} ± {row['Abs_Magnetization_Error']:.4f}",
                f"{row['Specific_Heat']:.4f} ± {row['Specific_Heat_Error']:.4f}",
                f"{row['Susceptibility']:.4f} ± {row['Susceptibility_Error']:.4f}"
            ])
        
        table = plt.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        pdf.savefig(fig)
        plt.close(fig)

# Main function
def main():
    data = read_data()
    
    ensemble_results = calculate_ensemble_averages(data)
    print("Ensemble Results:")
    print(ensemble_results)
    
    overall_results = calculate_overall_averages(ensemble_results)
    print("\nOverall Results:")
    print(overall_results)
    
    create_pdf_report(ensemble_results, overall_results, data)
    
    print("Analysis completed. Results saved to various plots and ising_model_report.pdf")

if __name__ == "__main__":
    main()