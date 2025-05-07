#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <cmath>

struct Coord {
    int x, y;
    bool operator<(const Coord& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
};

bool generate_sarw(std::vector<Coord>& path, int max_steps, std::mt19937& rng) {
    std::set<Coord> visited;
    path.clear();
    path.push_back({0, 0});
    visited.insert({0, 0});

    const Coord moves[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int step = 0; step < max_steps; ++step) {
        std::vector<Coord> available;
        Coord current = path.back();

        for (const auto& move : moves) {
            Coord next = {current.x + move.x, current.y + move.y};
            if (visited.find(next) == visited.end()) {
                available.push_back(next);
            }
        }

        if (available.empty()) {
            return false; // Trapped
        }

        std::uniform_int_distribution<int> dist(0, available.size() - 1);
        Coord next = available[dist(rng)];
        path.push_back(next);
        visited.insert(next);
    }
    return true;
}

double end_to_end_dist(const std::vector<Coord>& path) {
    if (path.empty()) return 0.0;
    const Coord& start = path.front();
    const Coord& end = path.back();
    return std::sqrt((end.x - start.x) * (end.x - start.x) + 
                     (end.y - start.y) * (end.y - start.y));
}

void write_path(const std::vector<Coord>& path, std::ofstream& file, int path_id) {
    for (size_t i = 0; i < path.size(); ++i) {
        file << path_id << " " << i << " " << path[i].x << " " << path[i].y << "\n";
    }
}

void write_final_config(const std::vector<Coord>& path, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing\n";
        return;
    }
    file << path.size() << "\n";
    file << "SARW final configuration\n";
    for (const auto& p : path) {
        file << p.x << " " << p.y << " 0.0\n";
    }
    file.close();
}

int main() {
    // Parameters
    const int max_steps = 100;
    const int num_paths = 1000;
    const int save_paths = 5;
    const int step_interval = 10;

    // Random number generator
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

    // Open output files
    std::ofstream path_file("sarw_positions.dat");
    std::ofstream scale_file("sarw_scaling.dat");
    if (!path_file.is_open() || !scale_file.is_open()) {
        std::cerr << "Error: Could not open output files\n";
        return 1;
    }

    std::vector<Coord> path;
    int completed_paths = 0;
    int path_id = 0;

    // Scaling data
    std::vector<double> r2_sum(max_steps + 1, 0.0);
    std::vector<int> path_count(max_steps + 1, 0);

    while (completed_paths < num_paths) {
        if (generate_sarw(path, max_steps, rng)) {
            completed_paths++;
            if (path_id < save_paths) {
                write_path(path, path_file, path_id);
            }

            // Compute R^2 for each step
            std::vector<Coord> sub_path;
            sub_path.push_back({0, 0});
            for (size_t i = 1; i < path.size(); ++i) {
                sub_path.push_back(path[i]);
                double r = end_to_end_dist(sub_path);
                r2_sum[i] += r * r;
                path_count[i]++;
            }

            // Save final configuration of last path
            if (completed_paths == num_paths) {
                write_final_config(path, "sarw_final_config.xyz");
            }

            path_id++;
        }
    }

    // Write scaling data
    for (int n = step_interval; n <= max_steps; n += step_interval) {
        if (path_count[n] > 0) {
            double avg_r2 = r2_sum[n] / path_count[n];
            scale_file << n << " " << avg_r2 << "\n";
        }
    }

    path_file.close();
    scale_file.close();

    std::cout << "Completed " << completed_paths 
              << " SARW simulations.\nFiles saved: sarw_positions.dat, sarw_scaling.dat, sarw_final_config.xyz\n";

    return 0;
}