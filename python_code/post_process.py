import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def load_data(folder_path, step, chemical):
    """ Load data from the specified chemical and time step. """
    filename = f'{folder_path}/combined_output_{chemical}_{step}.dat'
    return pd.read_csv(filename, sep='\t', header=None)

def animate(step):
    """ Update function for the animation. """
    ax1.clear()
    ax2.clear()

    # Load data for both chemicals at the current step
    data_c1 = load_data(folder_path, step, 'C1')
    data_c2 = load_data(folder_path, step, 'C2')

    # Create heatmaps
    c1 = ax1.imshow(data_c1, cmap='viridis', aspect='auto')
    c2 = ax2.imshow(data_c2, cmap='viridis', aspect='auto')

    # Add titles
    ax1.set_title('Chemical C1 at Step {}'.format(step))
    ax2.set_title('Chemical C2 at Step {}'.format(step))
    
    return c1, c2

# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Path to the folder containing data files
folder_path = '../build/out_combined'

# Detect how many steps there are based on the files in the folder for one of the chemicals
steps = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(folder_path) if 'C1' in f])
num_steps = len(steps)  # Update total number of steps

# Create animation
ani = FuncAnimation(fig, animate, frames=num_steps, repeat=False)

# Save the animation
ani.save('./animation_parallel.mp4', writer='ffmpeg', fps=10)

