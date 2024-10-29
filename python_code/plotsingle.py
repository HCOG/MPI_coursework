import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    """Load data from a .dat file assuming a simple space-separated format."""
    return pd.read_csv(filepath, delimiter='\t', header=None)  # Adjust delimiter as needed

def plot_heatmap(data):
    """Plot a heatmap from a 2D numpy array."""
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='viridis')  # Choose a colormap that fits your data and preference
    fig.colorbar(cax)
    plt.title("Heatmap of Data")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

def main():
    filepath = '../build/out/combined_output_C1.dat'  # Update with your .dat file path
    data = load_data(filepath)
    plot_heatmap(data.values)  # Ensure data is in a suitable 2D array format for plotting

if __name__ == "__main__":
    main()
