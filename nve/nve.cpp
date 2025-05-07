#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Structure to represent a 2D point/vector
struct Vec2 {
    double x, y;
    
    Vec2() : x(0.0), y(0.0) {}
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    
    // Vector addition
    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }
    
    // Vector subtraction
    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }
    
    // Scalar multiplication
    Vec2 operator*(double scalar) const {
        return Vec2(x * scalar, y * scalar);
    }
    
    // Magnitude squared
    double mag_squared() const {
        return x*x + y*y;
    }
    
    // Magnitude
    double mag() const {
        return std::sqrt(mag_squared());
    }
};

class MDSimulation {
private:
    // System parameters
    double L;              // Box size
    double rho;            // Number density
    int N;                 // Number of particles
    double dt;             // Time step
    int total_steps;       // Total simulation steps
    double r_cut;          // Cutoff radius for LJ potential
    double r_cut_sq;       // Square of cutoff radius
    bool use_gaussian;     // Flag for velocity initialization type
    
    // Particle data
    std::vector<Vec2> positions;
    std::vector<Vec2> velocities;
    std::vector<Vec2> forces;
    std::vector<Vec2> prev_forces; // For Velocity-Verlet integration
    
    // Energy and temperature data
    std::vector<double> potential_energy_data;
    std::vector<double> kinetic_energy_data;
    std::vector<double> total_energy_data;
    std::vector<double> temperature_data;
    std::vector<double> time_data;
    
    // For MSD calculation
    std::vector<Vec2> initial_positions;
    std::vector<double> msd_data;
    
    // Neighbor list
    struct NeighborList {
        std::vector<std::vector<int>> neighbors;
        std::vector<Vec2> ref_positions;
        double skin;
        double list_range_sq;
        bool needs_update;
        
        NeighborList(int n, double r_cut, double skin_factor = 0.3) : 
            neighbors(n), ref_positions(n), skin(r_cut * skin_factor), 
            list_range_sq(std::pow(r_cut + skin, 2)), needs_update(true) {}
    } neighbor_list;
    
    // Random number generator
    std::mt19937 rng;
    
public:
    MDSimulation(double L_, double rho_, double dt_, int total_steps_, bool use_gaussian_) :
        L(L_), rho(rho_), dt(dt_), total_steps(total_steps_), use_gaussian(use_gaussian_),
        r_cut(2.5), r_cut_sq(r_cut * r_cut), 
        neighbor_list(0, r_cut) {
        
        // Calculate number of particles based on density
        N = static_cast<int>(rho_ * L_ * L_);
        
        // Initialize vectors
        positions.resize(N);
        velocities.resize(N);
        forces.resize(N);
        prev_forces.resize(N);
        
        // Data vectors
        potential_energy_data.reserve(total_steps);
        kinetic_energy_data.reserve(total_steps);
        total_energy_data.reserve(total_steps);
        temperature_data.reserve(total_steps);
        time_data.reserve(total_steps);
        msd_data.reserve(total_steps);
        
        // Initialize neighbor list
        neighbor_list = NeighborList(N, r_cut);
        
        // Initialize random number generator
        std::random_device rd;
        rng = std::mt19937(rd());
        
        std::cout << "Initializing simulation with N = " << N << " particles" << std::endl;
    }
    
    // Place particles on a grid to avoid overlap
    void initialize_positions() {
        int side = static_cast<int>(std::ceil(std::sqrt(N)));
        double spacing = L / side;
        
        for (int i = 0; i < N; ++i) {
            int ix = i % side;
            int iy = i / side;
            
            // Add small random displacement to break symmetry
            std::uniform_real_distribution<double> small_disp(-0.1 * spacing, 0.1 * spacing);
            
            positions[i].x = (ix + 0.5) * spacing + small_disp(rng);
            positions[i].y = (iy + 0.5) * spacing + small_disp(rng);
            
            // Make sure particles stay within the box
            positions[i].x = std::fmod(positions[i].x + L, L);
            positions[i].y = std::fmod(positions[i].y + L, L);
        }
        
        // Store initial positions for MSD calculation
        initial_positions = positions;
    }
    
    void initialize_velocities() {
        double sum_vx = 0.0, sum_vy = 0.0;
        
        if (use_gaussian) {
            // Gaussian distribution with zero mean and unit variance
            std::normal_distribution<double> dist(0.0, 1.0);
            
            for (int i = 0; i < N; ++i) {
                velocities[i].x = dist(rng);
                velocities[i].y = dist(rng);
                sum_vx += velocities[i].x;
                sum_vy += velocities[i].y;
            }
        } else {
            // Uniform distribution in [-0.5, 0.5]
            std::uniform_real_distribution<double> dist(-0.5, 0.5);
            
            for (int i = 0; i < N; ++i) {
                velocities[i].x = dist(rng);
                velocities[i].y = dist(rng);
                sum_vx += velocities[i].x;
                sum_vy += velocities[i].y;
            }
        }
        
        // Remove center of mass motion
        double vx_cm = sum_vx / N;
        double vy_cm = sum_vy / N;
        
        for (int i = 0; i < N; ++i) {
            velocities[i].x -= vx_cm;
            velocities[i].y -= vy_cm;
        }
        
        // Scale velocities to set initial temperature
        double target_temp = 1.0;
        double current_temp = calculate_temperature();
        double scale_factor = std::sqrt(target_temp / current_temp);
        
        for (int i = 0; i < N; ++i) {
            velocities[i].x *= scale_factor;
            velocities[i].y *= scale_factor;
        }
    }
    
    // Calculate minimum image distance between two positions
    Vec2 minimum_image_vector(const Vec2& pos1, const Vec2& pos2) const {
        Vec2 dr = pos1 - pos2;
        
        // Apply periodic boundary conditions
        if (dr.x > 0.5 * L) dr.x -= L;
        else if (dr.x < -0.5 * L) dr.x += L;
        
        if (dr.y > 0.5 * L) dr.y -= L;
        else if (dr.y < -0.5 * L) dr.y += L;
        
        return dr;
    }
    
    // Update the neighbor list if necessary
    void update_neighbor_list() {
        bool needs_update = neighbor_list.needs_update;
        
        if (!needs_update && neighbor_list.ref_positions.size() == N) {
            // Check if any particle has moved more than half the skin distance
            double displacement_threshold = 0.25 * neighbor_list.skin * neighbor_list.skin;
            
            for (int i = 0; i < N; ++i) {
                Vec2 disp = minimum_image_vector(positions[i], neighbor_list.ref_positions[i]);
                if (disp.mag_squared() > displacement_threshold) {
                    needs_update = true;
                    break;
                }
            }
        } else {
            needs_update = true;
        }
        
        if (needs_update) {
            // Store reference positions
            neighbor_list.ref_positions = positions;
            
            // Reset neighbor lists
            for (int i = 0; i < N; ++i) {
                neighbor_list.neighbors[i].clear();
            }
            
            // Build neighbor lists
            for (int i = 0; i < N - 1; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    Vec2 rij = minimum_image_vector(positions[i], positions[j]);
                    double r_sq = rij.mag_squared();
                    
                    if (r_sq < neighbor_list.list_range_sq) {
                        neighbor_list.neighbors[i].push_back(j);
                        neighbor_list.neighbors[j].push_back(i);
                    }
                }
            }
            
            neighbor_list.needs_update = false;
            std::cout << "Updated neighbor list" << std::endl;
        }
    }
    
    // Calculate forces and potential energy using LJ potential
    double calculate_forces() {
        // Reset forces
        for (int i = 0; i < N; ++i) {
            forces[i] = Vec2(0.0, 0.0);
        }
        
        double potential = 0.0;
        
        // Use neighbor list for efficiency
        update_neighbor_list();
        
        for (int i = 0; i < N; ++i) {
            for (const int j : neighbor_list.neighbors[i]) {
                if (j > i) {  // Avoid double counting
                    Vec2 rij = minimum_image_vector(positions[i], positions[j]);
                    double r_sq = rij.mag_squared();
                    
                    if (r_sq < r_cut_sq) {
                        double r_2 = 1.0 / r_sq;
                        double r_6 = r_2 * r_2 * r_2;
                        double r_12 = r_6 * r_6;
                        
                        // Lennard-Jones force: F = 24ε*[(2/r^13) - (1/r^7)]*r
                        double force_mag = 24.0 * (2.0 * r_12 - r_6) * r_2;
                        Vec2 force_ij = rij * force_mag;
                        
                        forces[i] = forces[i] + force_ij;
                        forces[j] = forces[j] - force_ij;  // Newton's third law
                        
                        // Lennard-Jones potential: V = 4ε*[(σ/r)^12 - (σ/r)^6]
                        // with ε = 1, σ = 1
                        potential += 4.0 * (r_12 - r_6);
                    }
                }
            }
        }
        
        return potential;
    }
    
    // Calculate kinetic energy and temperature
    double calculate_kinetic_energy() const {
        double kinetic = 0.0;
        
        for (int i = 0; i < N; ++i) {
            double v_sq = velocities[i].mag_squared();
            kinetic += 0.5 * v_sq;
        }
        
        return kinetic;
    }
    
    double calculate_temperature() const {
        // T = (2 * K) / (N * d * k_B)
        // Where d = 2 (dimension), k_B = 1 (in reduced units)
        return calculate_kinetic_energy() / N;
    }
    
    // Calculate mean squared displacement
    double calculate_msd() const {
        double sum_sq_disp = 0.0;
        
        for (int i = 0; i < N; ++i) {
            Vec2 disp = minimum_image_vector(positions[i], initial_positions[i]);
            sum_sq_disp += disp.mag_squared();
        }
        
        return sum_sq_disp / N;
    }
    
    // Velocity-Verlet integration step
    void velocity_verlet_step() {
        // Store current forces for second half of velocity update
        prev_forces = forces;
        
        // Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*f(t)*dt^2
        for (int i = 0; i < N; ++i) {
            positions[i] = positions[i] + velocities[i] * dt + prev_forces[i] * (0.5 * dt * dt);
            
            // Apply periodic boundary conditions
            positions[i].x = std::fmod(positions[i].x + L, L);
            positions[i].y = std::fmod(positions[i].y + L, L);
        }
        
        // Calculate new forces f(t+dt)
        double potential = calculate_forces();
        
        // Update velocities: v(t+dt) = v(t) + 0.5*[f(t) + f(t+dt)]*dt
        for (int i = 0; i < N; ++i) {
            velocities[i] = velocities[i] + (prev_forces[i] + forces[i]) * (0.5 * dt);
        }
        
        // Calculate energies and temperature
        double kinetic = calculate_kinetic_energy();
        double temperature = calculate_temperature();
        double total_energy = potential + kinetic;
        
        // Save data
        potential_energy_data.push_back(potential / N);
        kinetic_energy_data.push_back(kinetic / N);
        total_energy_data.push_back(total_energy / N);
        temperature_data.push_back(temperature);
    }
    
    void equilibration(int steps) {
        std::cout << "Starting equilibration for " << steps << " steps..." << std::endl;
        
        for (int step = 0; step < steps; ++step) {
            velocity_verlet_step();
            
            if (step % 100 == 0) {
                std::cout << "Equilibration step " << step << ", T = " 
                          << temperature_data.back() << std::endl;
            }
        }
        
        // Clear any data collected during equilibration
        potential_energy_data.clear();
        kinetic_energy_data.clear();
        total_energy_data.clear();
        temperature_data.clear();
        time_data.clear();
        msd_data.clear();
        
        // Reset initial positions for MSD calculation
        initial_positions = positions;
        
        std::cout << "Equilibration completed." << std::endl;
    }
    
    void run() {
        std::cout << "Starting MD simulation for " << total_steps << " steps..." << std::endl;
        
        // Initial force calculation
        double potential = calculate_forces();
        double kinetic = calculate_kinetic_energy();
        double temperature = calculate_temperature();
        double total_energy = potential + kinetic;
        double msd = 0.0;
        
        // Save initial data
        potential_energy_data.push_back(potential / N);
        kinetic_energy_data.push_back(kinetic / N);
        total_energy_data.push_back(total_energy / N);
        temperature_data.push_back(temperature);
        time_data.push_back(0.0);
        msd_data.push_back(msd);
        
        for (int step = 1; step <= total_steps; ++step) {
            velocity_verlet_step();
            
            double current_time = step * dt;
            time_data.push_back(current_time);
            
            // Calculate MSD
            msd = calculate_msd();
            msd_data.push_back(msd);
            
            if (step % 1000 == 0) {
                std::cout << "Step " << step << "/" << total_steps 
                          << ", T = " << temperature_data.back() 
                          << ", E = " << total_energy_data.back() 
                          << ", MSD = " << msd << std::endl;
            }
        }
        
        std::cout << "Simulation completed." << std::endl;
    }
    
    void save_data(const std::string& prefix) {
        // Save energy data
        std::ofstream energy_file(prefix + "_energy.dat");
        energy_file << "# time potential_energy kinetic_energy total_energy temperature\n";
        
        for (size_t i = 0; i < time_data.size(); ++i) {
            energy_file << time_data[i] << " " 
                      << potential_energy_data[i] << " " 
                      << kinetic_energy_data[i] << " "
                      << total_energy_data[i] << " "
                      << temperature_data[i] << "\n";
        }
        energy_file.close();
        
        // Save MSD data
        std::ofstream msd_file(prefix + "_msd.dat");
        msd_file << "# time msd\n";
        
        for (size_t i = 0; i < time_data.size(); ++i) {
            msd_file << time_data[i] << " " << msd_data[i] << "\n";
        }
        msd_file.close();
        
        // Save final configuration
        std::ofstream config_file(prefix + "_final_config.xyz");
        config_file << N << "\n";
        config_file << "Final configuration\n";
        
        for (int i = 0; i < N; ++i) {
            config_file << "Ar " << positions[i].x << " " << positions[i].y << " 0.0\n";
        }
        config_file.close();
        
        std::cout << "Data saved with prefix: " << prefix << std::endl;
    }
};

int main() {
    // Simulation parameters
    double L = 50.0;                  // Box size
    double rho = 0.7;                 // Number density
    double dt = 0.001;                // Time step
    int total_steps = 100 / dt;       // Total simulation steps (t=100)
    int equilibration_steps = 10000;  // Equilibration steps
    bool use_gaussian = true;         // Use Gaussian velocity distribution
    
    // Create and run simulation
    MDSimulation md(L, rho, dt, total_steps, use_gaussian);
    md.initialize_positions();
    md.initialize_velocities();
    
    // Equilibrate the system
    md.equilibration(equilibration_steps);
    
    // Run the production simulation
    md.run();
    
    // Save data for later analysis
    md.save_data("lj_sim");
    
    return 0;
}