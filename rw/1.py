import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

def self_avoiding_walk(T=10000):
    """
    Simulates a self-avoiding random walk on a 2D square grid.
    
    Args:
        T (int): Total number of time steps
    
    Returns:
        tuple: (positions, success) where positions is a list of (x,y) coordinates 
        and success is a boolean indicating if the walk completed T steps
    """
    # Directions: right, up, left, down
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    # Initialize the current position and visited positions
    current_pos = (0, 0)
    positions = [current_pos]
    visited = {current_pos}
    
    # Perform the walk
    for _ in range(T):
        # Find valid neighbors (not visited)
        valid_moves = []
        for dx, dy in directions:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if next_pos not in visited:
                valid_moves.append(next_pos)
        
        # If no valid moves are available, the walk is trapped
        if not valid_moves:
            return positions, False
        
        # Choose a random valid move
        next_pos = valid_moves[np.random.randint(0, len(valid_moves))]
        
        # Update position and record it
        current_pos = next_pos
        positions.append(current_pos)
        visited.add(current_pos)
    
    return positions, True

def draw_trajectories(num_realisations=5, T=10000):
    """
    Draws trajectories for multiple realizations of the self-avoiding walk.
    
    Args:
        num_realisations (int): Number of independent realizations
        T (int): Total number of time steps
    
    Returns:
        list: List of trajectories (positions) for each realization
    """
    all_trajectories = []
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(num_realisations):
        positions, success = self_avoiding_walk(T)
        all_trajectories.append(positions)
        
        # Extract x and y coordinates for plotting
        x_coords, y_coords = zip(*positions)
        
        # Plot trajectory
        ax.plot(x_coords, y_coords, '-', label=f'Walk {i+1} (Steps: {len(positions)-1})')
        
        # Mark start and end points
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start' if i == 0 else "")
        ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='End' if i == 0 else "")
    
    ax.set_title(f'Self-Avoiding Random Walk Trajectories (T={T})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('trajectory_plot.png', dpi=300)
    plt.close()
    
    return all_trajectories

def calculate_msd(num_realisations=100, T=10000):
    """
    Calculates the mean square displacement (MSD) as a function of time.
    
    Args:
        num_realisations (int): Number of independent realizations
        T (int): Total number of time steps
    
    Returns:
        tuple: (time_points, msd_values)
    """
    # Store total square displacement for each time step
    total_squared_displacement = defaultdict(float)
    count_per_time = defaultdict(int)
    
    print(f"Calculating MSD with {num_realisations} realizations...")
    start_time = time.time()
    
    for i in range(num_realisations):
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"Completed {i}/{num_realisations} walks ({elapsed:.2f} seconds)")
        
        positions, success = self_avoiding_walk(T)
        
        # Calculate squared displacement for each time step
        origin = positions[0]
        for t, pos in enumerate(positions):
            dx = pos[0] - origin[0]
            dy = pos[1] - origin[1]
            total_squared_displacement[t] += dx**2 + dy**2
            count_per_time[t] += 1
    
    # Calculate MSD for each time step
    time_points = sorted(count_per_time.keys())
    msd_values = [total_squared_displacement[t] / count_per_time[t] for t in time_points]
    
    # Plot MSD
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, msd_values, 'b.-')
    plt.xlabel('Time Steps')
    plt.ylabel('Mean Square Displacement')
    plt.title('Mean Square Displacement vs Time')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('msd_plot.png', dpi=300)
    
    # Fit MSD with t^α
    log_time = np.log(time_points[1:])  # Skip t=0
    log_msd = np.log(msd_values[1:])    # Skip t=0
    
    # Linear regression on log-log data
    coef = np.polyfit(log_time, log_msd, 1)
    alpha = coef[0]
    
    # Plot fit
    plt.figure(figsize=(10, 6))
    plt.plot(time_points[1:], msd_values[1:], 'b.', label='Data')
    plt.plot(time_points[1:], np.exp(coef[1]) * np.power(time_points[1:], alpha), 'r-', 
             label=f'Fit: MSD ∝ t^{alpha:.4f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Mean Square Displacement')
    plt.title(f'MSD vs Time with Power Law Fit (α = {alpha:.4f})')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('msd_fit_plot.png', dpi=300)
    
    print(f"Alpha value: {alpha:.4f}")
    print(f"For normal random walk, α would be 1.0")
    
    return time_points, msd_values, alpha

def calculate_pdf(num_realisations=1000, time_points=[5, 50, 100, 1000, 5000, 10000]):
    """
    Calculates the probability distribution function (PDF) of positions at specified time points.
    
    Args:
        num_realisations (int): Number of independent realizations
        time_points (list): Time points at which to calculate the PDF
    
    Returns:
        dict: Dictionary mapping time points to position PDFs
    """
    position_data = {t: [] for t in time_points}
    max_time = max(time_points)
    
    print(f"Calculating PDF with {num_realisations} realizations...")
    start_time = time.time()
    
    for i in range(num_realisations):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"Completed {i}/{num_realisations} walks ({elapsed:.2f} seconds)")
            
        positions, success = self_avoiding_walk(max_time)
        
        if not success:
            # If the walk didn't complete, use the data we have
            for t in time_points:
                if t < len(positions):
                    position_data[t].append(positions[t])
        else:
            # If the walk completed successfully, use all data points
            for t in time_points:
                if t < len(positions):
                    position_data[t].append(positions[t])
    
    # Create PDFs and plot
    for t in time_points:
        if not position_data[t]:
            print(f"No data available for t={t}")
            continue
            
        positions = np.array(position_data[t])
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Create 2D histogram as a proper heatmap
        plt.figure(figsize=(10, 8))
        
        # Calculate the range for the grid
        max_range = max(abs(x_coords.max()), abs(x_coords.min()), 
                        abs(y_coords.max()), abs(y_coords.min())) + 5
        bin_size = max(1, int(max_range / 25))  # Adjust bin size based on range
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(
            x_coords, y_coords, 
            bins=[np.arange(-max_range, max_range+bin_size, bin_size),
                  np.arange(-max_range, max_range+bin_size, bin_size)],
            density=True
        )
        
        # Create heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(hist.T, origin='lower', aspect='equal', extent=extent, cmap='hot')
        cbar = plt.colorbar(label='Probability Density')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Position PDF Heatmap at t={t}')
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'pdf_heatmap_t{t}.png', dpi=300)
        plt.close()
        
        # Calculate radial distribution
        r = np.sqrt(x_coords**2 + y_coords**2)
        plt.figure(figsize=(8, 6))
        plt.hist(r, bins=30, density=True, alpha=0.7)
        plt.xlabel('Distance from Origin')
        plt.ylabel('Probability Density')
        plt.title(f'Radial PDF at t={t}')
        plt.grid(True)
        plt.savefig(f'radial_pdf_t{t}.png', dpi=300)
        plt.close()
    
    return position_data

def main():
    np.random.seed(42)  # For reproducibility
    
    # Part b: Draw trajectories
    print("Part b: Drawing trajectories...")
    trajectories = draw_trajectories(num_realisations=5)
    
    # Part c: Calculate MSD
    print("\nPart c: Calculating MSD...")
    time_points, msd_values, alpha = calculate_msd(num_realisations=100)
    
    # Part d: Calculate PDF
    print("\nPart d: Calculating PDF...")
    pdfs = calculate_pdf(num_realisations=1000)
    
    # Part e: Compare alpha with normal random walk
    print("\nPart e: Comparing with normal random walk...")
    print(f"Calculated alpha value: {alpha:.4f}")
    print(f"Normal random walk alpha: 1.0")
    print(f"Difference: {abs(alpha - 1.0):.4f}")
    
    if alpha < 1.0:
        print("The walk is sub-diffusive compared to normal random walk.")
    elif alpha > 1.0:
        print("The walk is super-diffusive compared to normal random walk.")
    else:
        print("The walk exhibits normal diffusion.")

if __name__ == "__main__":
    main()