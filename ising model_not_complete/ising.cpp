#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>

// Ising Model 2D Implementation with Metropolis Algorithm
class IsingModel {
private:
    int L;                      // Lattice size (LxL)
    std::vector<std::vector<int>> spins; // Spin configuration
    double J;                   // Exchange interaction
    double h;                   // External field
    double beta;                // Inverse temperature (1/kT)
    std::mt19937 rng;           // Random number generator
    
    // Get neighbor with periodic boundary conditions
    int periodic(int i, int L) {
        return (i + L) % L;
    }
    
    // Calculate energy contribution of a single spin
    double siteEnergy(int i, int j) {
        int sum = 0;
        // Sum over nearest neighbors
        sum += spins[periodic(i+1, L)][j];
        sum += spins[periodic(i-1, L)][j];
        sum += spins[i][periodic(j+1, L)];
        sum += spins[i][periodic(j-1, L)];
        
        // -J * s_i * sum_neighbors(s_j) - h * s_i
        return -J * spins[i][j] * sum - h * spins[i][j];
    }
    
    // Calculate total energy
    double calculateTotalEnergy() {
        double energy = 0.0;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                // Count only half the bonds to avoid double counting
                int s = spins[i][j];
                int sum = 0;
                sum += spins[periodic(i+1, L)][j];
                sum += spins[i][periodic(j+1, L)];
                energy += -J * s * sum;
                energy += -h * s;
            }
        }
        return energy;
    }
    
    // Calculate total magnetization
    double calculateMagnetization() {
        double mag = 0.0;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                mag += spins[i][j];
            }
        }
        return mag;
    }
    
public:
    // Constructor
    IsingModel(int lattice_size, double exchange, double field, double temperature, unsigned seed) 
        : L(lattice_size), J(exchange), h(field), beta(1.0/temperature), rng(seed) {
        // Initialize all spins up
        spins.resize(L, std::vector<int>(L, 1));
    }
    
    // Perform one Monte Carlo step (L^2 attempted spin flips)
    void mcStep() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<int> lattice_dist(0, L-1);
        
        for (int step = 0; step < L*L; step++) {
            // Randomly select a site
            int i = lattice_dist(rng);
            int j = lattice_dist(rng);
            
            // Calculate energy change if we flip this spin
            double oldEnergy = siteEnergy(i, j);
            spins[i][j] *= -1; // Flip spin
            double newEnergy = siteEnergy(i, j);
            double deltaE = newEnergy - oldEnergy;
            
            // Metropolis acceptance criterion
            if (deltaE > 0 && exp(-beta * deltaE) < dist(rng)) {
                // Reject the flip
                spins[i][j] *= -1; // Flip back
            }
        }
    }
    
    // Perform simulation
    std::vector<std::pair<double, double>> simulate(int equilibration_steps, int measurement_steps) {
        std::vector<std::pair<double, double>> results;
        
        // Equilibration phase
        for (int step = 0; step < equilibration_steps; step++) {
            mcStep();
        }
        
        // Measurement phase
        for (int step = 0; step < measurement_steps; step++) {
            mcStep();
            double energy = calculateTotalEnergy() / (L*L);  // Energy per site
            double mag = calculateMagnetization() / (L*L);   // Magnetization per site
            results.push_back({energy, mag});
        }
        
        return results;
    }
    
    // Save current spin configuration to file
    void saveConfiguration(const std::string &filename) {
        std::ofstream outfile(filename);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                outfile << spins[i][j] << " ";
            }
            outfile << "\n";
        }
        outfile.close();
    }
};

int main(int argc, char *argv[]) {
    // Parameters
    int L = 32;                   // Lattice size
    double J = 1.0;               // Exchange interaction
    double h = 0.0;               // External field
    double T = 1.0;               // Temperature
    int num_ensembles = 5;        // Number of independent ensembles
    int equilibration_steps = 1000; // Equilibration steps
    int measurement_steps = 10000;  // Measurement steps
    
    // Parse command line arguments
    if (argc > 1) T = std::stod(argv[1]);
    if (argc > 2) h = std::stod(argv[2]);
    if (argc > 3) L = std::stoi(argv[3]);
    if (argc > 4) num_ensembles = std::stoi(argv[4]);
    
    // Open output file
    std::ofstream datafile("ising_data.txt");
    datafile << "# Ensemble Temperature Energy Magnetization\n";
    
    // Run simulation for multiple independent ensembles
    for (int ensemble = 0; ensemble < num_ensembles; ensemble++) {
        // Use different seed for each ensemble
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() + ensemble;
        
        // Create and run simulation
        IsingModel model(L, J, h, T, seed);
        auto results = model.simulate(equilibration_steps, measurement_steps);
        
        // Save results
        for (const auto &result : results) {
            datafile << ensemble << " " << T << " " << result.first << " " << result.second << "\n";
        }
        
        // Save final configuration
        model.saveConfiguration("config_ensemble_" + std::to_string(ensemble) + ".txt");
    }
    
    datafile.close();
    std::cout << "Simulation completed. Data saved to ising_data.txt" << std::endl;
    
    return 0;
}