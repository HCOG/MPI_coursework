# Parallelization of coupled PDEs

This is part of my parallel programming coursework. In this project I will enable parallel programming for a coupled PDEs solver representing the reaction and diffusive transport of two chemical species.

## Installation

Follow these steps to get your development environment set up:

### Prerequisites

Ensure you have the following installed on your system:
- Git (Optional, for cloning the repository)
- Python 3
- MPI
- CMake
- Make
- ffmpeg

### Cloning the Repository

To clone the repository, run the following command in your terminal:

```bash
git clone https://github.com/HCOG/MPI_coursework.git
cd ppp-mpi-assessment-acse-zd1420
chmod +x run_mpi.sh
./run_mpi.sh
``````

## Project description
With the given serial code, I created a duplicate and started parallelizing it. I implemented a grid decomposition with an adaptive partitioning function so that with any given process number, it could partition the grid into the correct portion and allow the process to have the correct neighbours.

### Code Structure

The program is divided into several parts, focusing on grid partitioning, neighbor finding, simulation iterations, and I/O operations.

### Grid Partitioning

- **GridPartition Struct**: Holds the local grid boundaries and neighbor information for each process.
- **get_partition Function**: Calculates the grid partition for each process based on the total number of processes and the rank of the current process.
- **create_grid_partition_type Function**: Creates a custom MPI datatype for the `GridPartition` struct.

### Neighbor Determination

- **find_neighbors Function**: Identifies the neighboring partitions for each process. This function supports both Neumann and periodic boundary conditions.

### Iterative Calculation

- **do_iteration Function**: Performs a single simulation timestep. It includes data swapping, communication between processes, and computation of new values for the simulation variables based on reaction-diffusion dynamics.

### Input/Output Operations

- **setup_data_profile Function**: Initializes the simulation variables.
- **grid_to_file Function**: Outputs the grid data to files.
- **process_timesteps Function**: Manages the reading and writing of timestep data for partitioned output.
- **read_and_assemble Function**: Aggregates output data from different processes.

### Main Execution Flow

- Initializes MPI.
- Calculates grid partitions and sets up communication.
- Runs the simulation loop until a specified maximum time.
- Outputs data at predetermined intervals.
- Finalizes MPI and cleans up resources.


## Extension and additions
### Calculation during non-blocking communications:
During the communication sections of my code line(240-317), I started the calculation of the centre part of my processes's data that disregards boundary or neighbour conditions. The edges which do require neighbouring transmission, will only start updation after the MPI_Waitall.

### Combined C1 C2 during communication:
I joined C1 C2 data alternately in one single array, and pass this buffer for communication. Thus instead of 16 buffer arrays and 16 times sending and receiving operation, I only need 8 buffer arrays and communication operation.

### Parallelize post-processing:
The MPI-based approach for handling output and post-processing optimizes performance by distributing file I/O across multiple processes, thus minimizing bottlenecks and enhancing scalability. Each process manages its segment of data, which not only improves load balancing and reduces the risk of data corruption but also aligns well with the capabilities of parallel file systems.








