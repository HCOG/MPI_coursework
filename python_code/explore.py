import pandas as pd
import numpy as np

# Define the path to the .dat file
file_path = './build/out/combined_output_C1.dat'

# Load the data
# Adjust the delimiter based on your file's formatting
# Common delimiters include commas (','), tabs ('\t'), or spaces (' ')
df = pd.read_csv(file_path, delimiter='\t', header=None)  # Update delimiter and header as needed

# Display the shape of the dataframe
print("Dataframe Shape:")
print(df.shape)

# Display the first few rows of the dataframe
print("First 5 rows of the dataframe:")
print(df.head())