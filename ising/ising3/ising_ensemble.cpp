#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
using namespace std;

class IsingModel {
private:
    int L; // Lattice size (L x L)
    vector<vector<int>> lattice; // 2D lattice of spins
    double J; 
    double kB; 
    double h_z; 
    mt19937 rng; // Random number generator
    uniform_real_distribution<double> dist; 
    uniform_int_distribution<int> pos_dist; 
    unsigned long seed; // Random seed for reproducibility

public:
    // Constructor
    IsingModel(int size, double exchange, double boltzmann, double field, unsigned long custom_seed) 
        : L(size), J(exchange), kB(boltzmann), h_z(field), 
          lattice(size, vector<int>(size, 1)), // Initializing all spins up
          dist(0.0, 1.0), pos_dist(0, size-1), seed(custom_seed) {
        
        rng.seed(seed);
    }

    // Calculating energy change for a spin flip at position (i,j)
    double calculateDeltaE(int i, int j) {
        int spin = lattice[i][j];
        int sum_neighbors = 0;
        
        sum_neighbors += lattice[(i+1) % L][j];       // R
        sum_neighbors += lattice[(i-1+L) % L][j];     // L
        sum_neighbors += lattice[i][(j+1) % L];       // U
        sum_neighbors += lattice[i][(j-1+L) % L];     // D
        
        // Delta E = 2*J*s_i*sum_neighbors + 2*h_z*s_i
        return 2.0 * J * spin * sum_neighbors + 2.0 * h_z * spin;
    }

    // Calculating total energy of the system
    double calculateEnergy() {
        double energy = 0.0;
        
        // Sum over all lattice sites
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                if (j < L-1) energy -= J * lattice[i][j] * lattice[i][j+1];
                if (i < L-1) energy -= J * lattice[i][j] * lattice[i+1][j];
                
                energy -= h_z * lattice[i][j];
            }
        }
        
        return energy;
    }

    // Calculating magnetization per spin
    double calculateMagnetization() {
        double M = 0.0;
        
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < L; ++j) {
                M += lattice[i][j];
            }
        }
        
        return M / (L * L);
    }

    // Performing a single Monte Carlo step (sweep)
    void mcStep(double T) {
        for (int step = 0; step < L * L; ++step) {
            // Randomly selecting a site
            int i = pos_dist(rng);
            int j = pos_dist(rng);
            
            double deltaE = calculateDeltaE(i, j);
            
            // Metropolis acceptance criterion
            if (deltaE <= 0.0 || dist(rng) < exp(-deltaE / (kB * T))) {
                lattice[i][j] *= -1; // Flipping the spin
            }
        }
    }

    pair<double, double> simulateAtTemperature(double T, int equilibration_steps, int production_steps) {
        // Equilibration phase
        for (int step = 0; step < equilibration_steps; ++step) {
            mcStep(T);
        }
        
        // Production phase
        double M_avg = 0.0;
        double M2_avg = 0.0;
        
        for (int step = 0; step < production_steps; ++step) {
            mcStep(T);
            double M = calculateMagnetization();
            double abs_M = abs(M);
            
            M_avg += abs_M;
            M2_avg += M * M;
        }
        
        M_avg /= production_steps;
        M2_avg /= production_steps;
        
        // Calculate susceptibility: χ = (⟨M²⟩ - ⟨|M|⟩²) / (k_B * T)
        double susceptibility = (M2_avg - M_avg * M_avg) * L * L / (kB * T);
        
        return {M_avg, susceptibility};
    }
};

void runEnsembles(int L, double J, double kB, double h_z, 
                 double T_start, double T_end, double T_step,
                 int n_ensembles, int equilibration_steps, int production_steps,
                 const string& filename) {
    
    // Prepare output file
    ofstream output(filename);
    output << "Temperature,AbsoluteMagnetization,MagnetizationError,Susceptibility,SusceptibilityError\n";
    
    // Loop over temperatures
    for (double T = T_start; T <= T_end + T_step; T += T_step) {
        cout << "Simulating T = " << T << endl;
        
        vector<double> magnetizations(n_ensembles);
        vector<double> susceptibilities(n_ensembles);
        
        for (int e = 0; e < n_ensembles; ++e) {
            cout << "  Ensemble " << (e+1) << "/" << n_ensembles << endl;
            
            unsigned long seed = 12345 + e;
            IsingModel model(L, J, kB, h_z, seed);
            
            auto [M_avg, susceptibility] = model.simulateAtTemperature(T, equilibration_steps, production_steps);
            
            magnetizations[e] = M_avg;
            susceptibilities[e] = susceptibility;
        }
        
        double M_ensemble_avg = 0.0;
        double chi_ensemble_avg = 0.0;
        
        for (int e = 0; e < n_ensembles; ++e) {
            M_ensemble_avg += magnetizations[e];
            chi_ensemble_avg += susceptibilities[e];
        }
        
        M_ensemble_avg /= n_ensembles;
        chi_ensemble_avg /= n_ensembles;
        
        double M_error = 0.0;
        double chi_error = 0.0;
        
        for (int e = 0; e < n_ensembles; ++e) {
            M_error += (magnetizations[e] - M_ensemble_avg) * (magnetizations[e] - M_ensemble_avg);
            chi_error += (susceptibilities[e] - chi_ensemble_avg) * (susceptibilities[e] - chi_ensemble_avg);
        }
        
        M_error = sqrt(M_error / (n_ensembles * (n_ensembles - 1)));
        chi_error = sqrt(chi_error / (n_ensembles * (n_ensembles - 1)));
        
        output << fixed << setprecision(6)
               << T << "," 
               << M_ensemble_avg << "," 
               << M_error << "," 
               << chi_ensemble_avg << "," 
               << chi_error << endl;
    }
    
    output.close();
}

int main(int argc, char* argv[]) {
    double h_z = 0.0;
    if (argc > 1) h_z = stod(argv[1]);
    
    int L = 40;  
    if (argc > 2) L = stoi(argv[2]);
    
    // Parameters
    double J = 1.0;           
    double kB = 1.0;          
    double T_start = 0.1;     
    double T_end = 3.0;       
    double T_step = 0.1;      
    int n_ensembles = 5;      // Number of independent ensembles
    int equilibration_steps = 1000;  // Equilibration steps per temperature
    int production_steps = 5000;    // Production steps per temperature
    
    string filename = "ising_L" + to_string(L) + "_h" + to_string(h_z) + "_ensemble.csv";
    
    runEnsembles(L, J, kB, h_z, T_start, T_end, T_step, 
                n_ensembles, equilibration_steps, production_steps, filename);
    
    cout << "Simulation completed. Results saved to " << filename << endl;
    
    return 0;
}