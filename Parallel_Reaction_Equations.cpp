#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <filesystem>

using namespace std;

// Define a struct to hold the partition of the grid for each process
struct GridPartition {
    int istart, iend, jstart, jend, local_imax, local_jmax;
	int north, south, east, west;
	int rank;
};

// Function to determine the partition of the grid for each process
GridPartition get_partition(int imax, int jmax, int num_procs, int rank) {
    // Determine the number of partitions along each dimension
    int p_rows = sqrt(num_procs);
    while (num_procs % p_rows != 0) {
        p_rows--;
    }
    int p_cols = num_procs / p_rows;

    // Size of each partition
    int block_i_size = imax / p_rows;
    int block_j_size = jmax / p_cols;

    // Determine the row and column of the current process
    int proc_row = rank / p_cols;
    int proc_col = rank % p_cols;

    // Calculate the local indices for this process
    GridPartition partition;
    partition.istart = proc_row * block_i_size;
    partition.iend = (proc_row == p_rows - 1) ? imax : (proc_row + 1) * block_i_size;
    partition.jstart = proc_col * block_j_size;
    partition.jend = (proc_col == p_cols - 1) ? jmax : (proc_col + 1) * block_j_size;
	partition.local_imax = partition.iend - partition.istart;
	partition.local_jmax = partition.jend - partition.jstart;
	partition.rank = rank;

    return partition;
}

/*A function that creates a custom made MPI_datatype that can convey the GridPartition.
This is crucial information that need to be shared across processes so that during
later stage of post-processing, each process could combine the output files from all
other processes.*/
MPI_Datatype create_grid_partition_type() {
    MPI_Datatype grid_partition_type;
    // Update blocklengths array to include additional entries for the neighbor fields
    int blocklengths[11] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};  // One block for each integer field
    MPI_Aint offsets[11];
    // Add offsets for each field in the struct
    offsets[0] = offsetof(GridPartition, istart);
    offsets[1] = offsetof(GridPartition, iend);
    offsets[2] = offsetof(GridPartition, jstart);
    offsets[3] = offsetof(GridPartition, jend);
    offsets[4] = offsetof(GridPartition, local_imax);
    offsets[5] = offsetof(GridPartition, local_jmax);
    offsets[6] = offsetof(GridPartition, north);
    offsets[7] = offsetof(GridPartition, south);
    offsets[8] = offsetof(GridPartition, east);
    offsets[9] = offsetof(GridPartition, west);
	offsets[10] = offsetof(GridPartition, rank);

    MPI_Datatype types[11] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    MPI_Type_create_struct(11, blocklengths, offsets, types, &grid_partition_type);
    MPI_Type_commit(&grid_partition_type);

    return grid_partition_type;
}

// Function to determine neighbors given a rank and a vector of partitions
void find_neighbors(int rank, const vector<GridPartition>& partitions, GridPartition& currentPartition, bool isPeriodic) {
    int num_procs = partitions.size();
    int p_rows = sqrt(num_procs);
    while (num_procs % p_rows != 0) {
        p_rows--;
    }
    int p_cols = num_procs / p_rows;

    // Initialize neighbors to -1 (indicating boundaries for Neumann condition)
    currentPartition.north = -1;
    currentPartition.south = -1;
    currentPartition.east = -1;
    currentPartition.west = -1;
    currentPartition.rank = rank;

    int proc_row = rank / p_cols;
    int proc_col = rank % p_cols;

    // Check and set the neighboring ranks
    if (proc_row > 0) {
        currentPartition.north = rank - p_cols;
    } else if (isPeriodic) {
        currentPartition.north = rank + p_cols * (p_rows - 1);
    }

    if (proc_row < p_rows - 1) {
        currentPartition.south = rank + p_cols;
    } else if (isPeriodic) {
        currentPartition.south = rank - p_cols * (p_rows - 1);
    }

    if (proc_col > 0) {
        currentPartition.west = rank - 1;
    } else if (isPeriodic) {
        currentPartition.west = rank + (p_cols - 1);
    }

    if (proc_col < p_cols - 1) {
        currentPartition.east = rank + 1;
    } else if (isPeriodic) {
        currentPartition.east = rank - (p_cols - 1);
    }
}

/* Reads in the global matrix C1 and C2, and for each existing processes, load the corresponding files,
and files the partitioning information from the partition vector*/
void read_and_assemble(vector<double>& global_matrix_C1, vector<double>& global_matrix_C2, vector<GridPartition>& partitions, int imax, int jmax, int num_procs, int timestep) {
    ifstream infile_C1, infile_C2;
    int proc_id;

    for (proc_id = 0; proc_id < num_procs; proc_id++) {
        // Open C1 file
        stringstream filename_C1;
        filename_C1 << "./out/output_C1_" << timestep << "_" << proc_id << ".dat";
        infile_C1.open(filename_C1.str());

        // Open C2 file
        stringstream filename_C2;
        filename_C2 << "./out/output_C2_" << timestep << "_" << proc_id << ".dat";
        infile_C2.open(filename_C2.str());

        if (!infile_C1.is_open() || !infile_C2.is_open()) {
            cout << "Failed to open files for process " << proc_id << " at timestep " << timestep << endl;
            continue;
        }

        GridPartition part = partitions[proc_id];

        for (int i = 0; i < part.local_imax; i++) {
            for (int j = 0; j < part.local_jmax; j++) {
                double value_C1, value_C2;
                int global_index = (part.istart + i) * jmax + (part.jstart + j);

                infile_C1 >> value_C1;
                global_matrix_C1[global_index] = value_C1;

                infile_C2 >> value_C2;
                global_matrix_C2[global_index] = value_C2;
            }
        }

        infile_C1.close();
        infile_C2.close();
    }
}

/* Given the start and end timestep range calculated for each processes, the function
will allow a process to only combine outputs from a certain range.*/
void process_timesteps(int start_timestep, int end_timestep, int imax, int jmax, vector<GridPartition>& partitions, int num_procs, int id) {
    
    vector<double> global_matrix_C1(imax * jmax), global_matrix_C2(imax * jmax);

    for (int t = start_timestep; t < end_timestep; t++) {
        // Process each timestep assigned to this MPI process
        read_and_assemble(global_matrix_C1, global_matrix_C2, partitions, imax, jmax, num_procs, t);

        // Output C1
        ofstream outfile_C1("./out_combined/combined_output_C1_" + to_string(t) + ".dat");
        for (int i = 0; i < imax; i++) {
            for (int j = 0; j < jmax; j++) {
                outfile_C1 << global_matrix_C1[i * jmax + j] << "\t";
            }
            outfile_C1 << endl;
        }
        outfile_C1.close();
        // Output C2
        ofstream outfile_C2("./out_combined/combined_output_C2_" + to_string(t) + ".dat");
        for (int i = 0; i < imax; i++) {
            for (int j = 0; j < jmax; j++) {
                outfile_C2 << global_matrix_C2[i * jmax + j] << "\t";
            }
            outfile_C2 << endl;
        }
        outfile_C2.close();
		cout<< "Finished assembling timestep " << t << endl;
		cout.flush();
    }
}

///////////////////////////////////////////////////////////////////////////////////////
//////////// initiate global variables and constants for the simulation   /////////////
///////////////////////////////////////////////////////////////////////////////////////
double* C1, *C1_old, *C2, *C2_old;

int imax = 301, jmax = 301;
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.1, dt;
double y_max = 30.0, x_max = 30.0, dx, dy;

//set up simulation constants
const double f = 2.0, q = 0.002, epsilon = 0.03, D1 = 1.0, D2 = 0.6;


//Do a single time step
void do_iteration(GridPartition part, double* send_buffer_north, double* send_buffer_south, double* send_buffer_east, double* send_buffer_west, double* recv_buffer_north, double* recv_buffer_south, double* recv_buffer_east, double* recv_buffer_west)
{
	//Note that I am not copying data between the grids, which would be very slow, but rather just swapping pointers
	swap(C1, C1_old);
	swap(C2, C2_old);


	///////////////////////////////////////////////////////////////////////////////////////
	//////////////     Sending and receiving data between processes     //////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	// Copy data to send buffers
	// Copy data to send buffers
	for (int j = 0; j < part.local_jmax; j++) {
		send_buffer_north[2 * j] = C1_old[j]; // Properly index to handle double data size
		send_buffer_north[2 * j + 1] = C2_old[j]; // Second half of the data for C2

		send_buffer_south[2 * j] = C1_old[(part.local_imax - 1) * part.local_jmax + j]; // Accessing the last row for the southern buffer
		send_buffer_south[2 * j + 1] = C2_old[(part.local_imax - 1) * part.local_jmax + j];
	}

	for (int i = 0; i < part.local_imax; i++) {
		send_buffer_east[i*2] = C1_old[i * part.local_jmax + part.local_jmax - 1];
		send_buffer_east[i*2+1] = C2_old[i * part.local_jmax + part.local_jmax - 1];
		send_buffer_west[i*2] = C1_old[i * part.local_jmax];
		send_buffer_west[i*2+1] = C2_old[i * part.local_jmax];
	}
	
	// Send and receive data
	MPI_Request requests[8];
	int rcount = 0;
	// Initialize the request array to prevent using uninitialized requests.
	for (int i = 0; i < 8; ++i) {
		requests[i] = MPI_REQUEST_NULL;
	}

	if (part.north != -1) {
		MPI_Isend(send_buffer_north, part.local_jmax*2, MPI_DOUBLE, part.north, 0, MPI_COMM_WORLD, &requests[rcount++]);
		MPI_Irecv(recv_buffer_north, part.local_jmax*2, MPI_DOUBLE, part.north, 2, MPI_COMM_WORLD, &requests[rcount++]);
	}

	if (part.south != -1) {
		MPI_Isend(send_buffer_south, part.local_jmax*2, MPI_DOUBLE, part.south, 2, MPI_COMM_WORLD, &requests[rcount++]);
		MPI_Irecv(recv_buffer_south, part.local_jmax*2, MPI_DOUBLE, part.south, 0, MPI_COMM_WORLD, &requests[rcount++]);
	}

	if (part.east != -1) {
		MPI_Isend(send_buffer_east, part.local_imax*2, MPI_DOUBLE, part.east, 1, MPI_COMM_WORLD, &requests[rcount++]);
		MPI_Irecv(recv_buffer_east, part.local_imax*2, MPI_DOUBLE, part.east, 3, MPI_COMM_WORLD, &requests[rcount++]);
	}
	

	if (part.west != -1) {
		MPI_Isend(send_buffer_west, part.local_imax*2, MPI_DOUBLE, part.west, 3, MPI_COMM_WORLD, &requests[rcount++]);
		MPI_Irecv(recv_buffer_west, part.local_imax*2, MPI_DOUBLE, part.west, 1, MPI_COMM_WORLD, &requests[rcount++]);
	}

	
	///////////////////////////////////////////////////////////////////////////////////////
	//////////////    Calculating the new concentrations for all points    ////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	
	/* As we are now using non-blocking communication, while we wait for all communication
	to finish, we can start calculation of center calculation for all other points. Doing
	this way also helps me to isolate out the edges where I need to determine a communication
	or use of boundary condition. */

	for (int i = 1; i < part.local_imax - 1; i++) {
		for (int j = 1; j < part.local_jmax - 1; j++) {
			// Compute the 1D array index for the 2D grid position (i, j)
			int idx = i * (part.local_jmax) + j;

			// Calculate the diffusion term for C1
			double diffusionC1 = D1 * ((C1_old[(i+1) * part.local_jmax + j] + C1_old[(i-1) * part.local_jmax + j] - 2.0 * C1_old[idx]) / (dx * dx) 
									+ (C1_old[idx + 1] + C1_old[idx - 1] - 2.0 * C1_old[idx]) / (dy * dy));

			// Calculate the reaction term for C1
			double reactionC1 = (C1_old[idx] * (1.0 - C1_old[idx]) - f * C2_old[idx] * (C1_old[idx] - q) / (C1_old[idx] + q)) / epsilon;

			// Update C1 based on old value, reaction, and diffusion
			C1[idx] = C1_old[idx] + dt * (reactionC1 + diffusionC1);

			// Calculate the diffusion term for C2
			double diffusionC2 = D2 * ((C2_old[(i+1) * part.local_jmax + j] + C2_old[(i-1) * part.local_jmax + j] - 2.0 * C2_old[idx]) / (dx * dx)
									+ (C2_old[idx + 1] + C2_old[idx - 1] - 2.0 * C2_old[idx]) / (dy * dy));

			// Calculate the reaction term for C2
			double reactionC2 = C1_old[idx] - C2_old[idx];

			// Update C2 based on old value, reaction, and diffusion
			C2[idx] = C2_old[idx] + dt * (reactionC2 + diffusionC2);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////
	//////////////                   Calculate the edges 				 ////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	/* For the edges, I will first disregards if it is at boundary and proceed with the 
	normal calculation. First I will calculate the four corners in the grid as they are
	involved with calculation with two buffers. Then I will proceed with the four edges
	which only envolves calculation with one buffers each. Lastly, I will do a boundary
	detection to override any rows or columns that were actually at boundary. */

	// Wait for all communication to finish
	MPI_Waitall(rcount, requests, MPI_STATUSES_IGNORE);
	
	// top-left index
	int tl_idx = 0; 
	C1[tl_idx] = C1_old[tl_idx] + dt * ((C1_old[tl_idx] * (1.0 - C1_old[tl_idx]) - f * C2_old[tl_idx] * (C1_old[tl_idx] - q) / (C1_old[tl_idx] + q)) / epsilon
    	+ D1 * ((recv_buffer_west[0] + C1_old[1] - 2.0 * C1_old[tl_idx]) / (dx * dx) + (recv_buffer_north[0] + C1_old[part.local_jmax] - 2.0 * C1_old[tl_idx]) / (dy * dy)));

	C2[tl_idx] = C2_old[tl_idx] + dt * (C1_old[tl_idx] - C2_old[tl_idx]
    	+ D2 * ((recv_buffer_west[1] + C2_old[1] - 2.0 * C2_old[tl_idx]) / (dx * dx) + (recv_buffer_north[1] + C2_old[part.local_jmax] - 2.0 * C2_old[tl_idx]) / (dy * dy)));


	// Top-right corner
	int tr_idx = part.local_jmax - 1; // Index of the top-right corner
	C1[tr_idx] = C1_old[tr_idx] + dt * ((C1_old[tr_idx] * (1.0 - C1_old[tr_idx]) - f * C2_old[tr_idx] * (C1_old[tr_idx] - q) / (C1_old[tr_idx] + q)) / epsilon
	    + D1 * ((C1_old[tr_idx - 1] + recv_buffer_east[0] - 2.0 * C1_old[tr_idx]) / (dx * dx) + (recv_buffer_north[part.local_jmax * 2 - 2] + C1_old[tr_idx + part.local_jmax] - 2.0 * C1_old[tr_idx]) / (dy * dy)));

	C2[tr_idx] = C2_old[tr_idx] + dt * (C1_old[tr_idx] - C2_old[tr_idx]
	    + D2 * ((C2_old[tr_idx - 1] + recv_buffer_east[1] - 2.0 * C2_old[tr_idx]) / (dx * dx) + (recv_buffer_north[part.local_jmax * 2 - 1] + C2_old[tr_idx + part.local_jmax] - 2.0 * C2_old[tr_idx]) / (dy * dy)));


	// Bottom-left corner
	int bl_idx = (part.local_imax - 1) * part.local_jmax; // Index of the bottom-left corner
	C1[bl_idx] = C1_old[bl_idx] + dt * ((C1_old[bl_idx] * (1.0 - C1_old[bl_idx]) - f * C2_old[bl_idx] * (C1_old[bl_idx] - q) / (C1_old[bl_idx] + q)) / epsilon
	    + D1 * ((recv_buffer_west[part.local_imax * 2 - 2] + C1_old[bl_idx + 1] - 2.0 * C1_old[bl_idx]) / (dx * dx) + (recv_buffer_south[0] + C1_old[bl_idx - part.local_jmax] - 2.0 * C1_old[bl_idx]) / (dy * dy)));

	C2[bl_idx] = C2_old[bl_idx] + dt * (C1_old[bl_idx] - C2_old[bl_idx]
	    + D2 * ((recv_buffer_west[part.local_imax * 2 - 1] + C2_old[bl_idx + 1] - 2.0 * C2_old[bl_idx]) / (dx * dx) + (recv_buffer_south[1] + C2_old[bl_idx - part.local_jmax] - 2.0 * C2_old[bl_idx]) / (dy * dy)));

	// Bottom-right corner
	int br_idx = part.local_imax * part.local_jmax - 1; // Index of the bottom-right corner
	C1[br_idx] = C1_old[br_idx] + dt * ((C1_old[br_idx] * (1.0 - C1_old[br_idx]) - f * C2_old[br_idx] * (C1_old[br_idx] - q) / (C1_old[br_idx] + q)) / epsilon
	    + D1 * ((C1_old[br_idx - 1] + recv_buffer_east[part.local_imax * 2 - 2] - 2.0 * C1_old[br_idx]) / (dx * dx) + (C1_old[br_idx - part.local_jmax] + recv_buffer_south[part.local_jmax * 2 - 2] - 2.0 * C1_old[br_idx]) / (dy * dy)));

	C2[br_idx] = C2_old[br_idx] + dt * (C1_old[br_idx] - C2_old[br_idx]
	    + D2 * ((C2_old[br_idx - 1] + recv_buffer_east[part.local_imax * 2 - 1] - 2.0 * C2_old[br_idx]) / (dx * dx) + (C2_old[br_idx - part.local_jmax] + recv_buffer_south[part.local_jmax * 2 - 1] - 2.0 * C2_old[br_idx]) / (dy * dy)));

	// Update the four edges

	// Update the top edge (excluding corners)
	// Update the top edge (excluding corners)
	int top_start = 0;  // Starting index of the top row
	for (int j = 1; j < part.local_jmax - 1; j++) {
		int idx = top_start + j;  // Index in the top row, excluding corners
		C1[idx] = C1_old[idx] + dt * ((C1_old[idx] * (1.0 - C1_old[idx]) - f * C2_old[idx] * (C1_old[idx] - q) / (C1_old[idx] + q)) / epsilon
			+ D1 * ((C1_old[idx - 1] + C1_old[idx + 1] - 2.0 * C1_old[idx]) / (dx * dx) + (recv_buffer_north[j*2] + C1_old[idx + part.local_jmax] - 2.0 * C1_old[idx]) / (dy * dy)));

		C2[idx] = C2_old[idx] + dt * (C1_old[idx] - C2_old[idx]
			+ D2 * ((C2_old[idx - 1] + C2_old[idx + 1] - 2.0 * C2_old[idx]) / (dx * dx) + (recv_buffer_north[j*2 + 1] + C2_old[idx + part.local_jmax] - 2.0 * C2_old[idx]) / (dy * dy)));
	}

	// Update the bottom edge (excluding corners)
	int bottom_start = (part.local_imax - 1) * part.local_jmax;
	for (int j = 1; j < part.local_jmax - 1; j++) {
		int idx = bottom_start + j;  // index in the bottom row, excluding corners
		C1[idx] = C1_old[idx] + dt * ((C1_old[idx] * (1.0 - C1_old[idx]) - f * C2_old[idx] * (C1_old[idx] - q) / (C1_old[idx] + q)) / epsilon
			+ D1 * ((C1_old[idx - 1] + C1_old[idx + 1] - 2.0 * C1_old[idx]) / (dx * dx) + (C1_old[idx - part.local_jmax] + recv_buffer_south[j*2] - 2.0 * C1_old[idx]) / (dy * dy)));

		C2[idx] = C2_old[idx] + dt * (C1_old[idx] - C2_old[idx]
			+ D2 * ((C2_old[idx - 1] + C2_old[idx + 1] - 2.0 * C2_old[idx]) / (dx * dx) + (C2_old[idx - part.local_jmax] + recv_buffer_south[j*2 + 1] - 2.0 * C2_old[idx]) / (dy * dy)));
	}


	// Update the left edge (excluding corners)
	for (int i = 1; i < part.local_imax - 1; i++) {
		int idx = i * part.local_jmax;  // index at the start of each row, excluding corners
		C1[idx] = C1_old[idx] + dt * ((C1_old[idx] * (1.0 - C1_old[idx]) - f * C2_old[idx] * (C1_old[idx] - q) / (C1_old[idx] + q)) / epsilon
			+ D1 * ((recv_buffer_west[i*2] + C1_old[idx + 1] - 2.0 * C1_old[idx]) / (dx * dx) + (C1_old[idx - part.local_jmax] + C1_old[idx + part.local_jmax] - 2.0 * C1_old[idx]) / (dy * dy)));

		C2[idx] = C2_old[idx] + dt * (C1_old[idx] - C2_old[idx]
			+ D2 * ((recv_buffer_west[i*2 + 1] + C2_old[idx + 1] - 2.0 * C2_old[idx]) / (dx * dx) + (C2_old[idx - part.local_jmax] + C2_old[idx + part.local_jmax] - 2.0 * C2_old[idx]) / (dy * dy)));
	}

	// Update the right edge (excluding corners)
	for (int i = 1; i < part.local_imax - 1; i++) {
		int idx = i * part.local_jmax + (part.local_jmax - 1);  // index at the end of each row, excluding corners
		C1[idx] = C1_old[idx] + dt * ((C1_old[idx] * (1.0 - C1_old[idx]) - f * C2_old[idx] * (C1_old[idx] - q) / (C1_old[idx] + q)) / epsilon
			+ D1 * ((C1_old[idx - 1] + recv_buffer_east[i*2] - 2.0 * C1_old[idx]) / (dx * dx) + (C1_old[idx - part.local_jmax] + C1_old[idx + part.local_jmax] - 2.0 * C1_old[idx]) / (dy * dy)));

		C2[idx] = C2_old[idx] + dt * (C1_old[idx] - C2_old[idx]
			+ D2 * ((C2_old[idx - 1] + recv_buffer_east[i*2 + 1] - 2.0 * C2_old[idx]) / (dx * dx) + (C2_old[idx - part.local_jmax] + C2_old[idx + part.local_jmax] - 2.0 * C2_old[idx]) / (dy * dy)));
	}

	///////////////////////////////////////////////////////////////////////////////////////
	//////////////    Boundary detection and override for edges and corners    ////////////
	///////////////////////////////////////////////////////////////////////////////////////

	// If we are dealing with periodic boundary conditions, this section of code will be skipped, providing we have the correct neighbors
	if (part.north == -1) {
		for (int j = 0; j < part.local_jmax; j++) {
			C1[j] = C1[part.local_jmax + j];
			C2[j] = C2[part.local_jmax + j];
		}
	}

	if (part.south == -1) {
		for (int j = 0; j < part.local_jmax; j++) {
			C1[(part.local_imax - 1) * part.local_jmax + j] = C1[(part.local_imax - 2) * part.local_jmax + j];
			C2[(part.local_imax - 1) * part.local_jmax + j] = C2[(part.local_imax - 2) * part.local_jmax + j];
		}
	}

	if (part.east == -1) {
		for (int i = 0; i < part.local_imax; i++) {
			C1[i * part.local_jmax + part.local_jmax - 1] = C1[i * part.local_jmax + part.local_jmax - 2];
			C2[i * part.local_jmax + part.local_jmax - 1] = C2[i * part.local_jmax + part.local_jmax - 2];
		}
	}

	if (part.west == -1) {
		for (int i = 0; i < part.local_imax; i++) {
			C1[i * part.local_jmax] = C1[i * part.local_jmax + 1];
			C2[i * part.local_jmax] = C2[i * part.local_jmax + 1];
		}
	}

	t += dt;
}


void calc_constants()
{
	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)imax - 1);

	t = 0.0;

	dt = 0.1 * pow(min(dx, dy), 2.0) / (2.0 * max(D1, D2));
}

/* I modified the setup data profile part, where I initialize all the C1
C2, C1_old, and C2_old as 1D double array. It also takes in the GridPartition
struct to access the start-end information for each processes.*/
void setup_data_profile(GridPartition part, int id) {
    double origin_x = x_max / 2.0, origin_y = y_max / 2.0;

	int size = part.local_imax * part.local_jmax;  // Total size of each array
	C1 = new double[size];
	C1_old = new double[size];
	C2 = new double[size];
	C2_old = new double[size];
	if (id == 0) {
		cout<< "Size of C1: " << size << endl;
		cout<< "Size of C2: " << size << endl;

		cout<< "Local imax: " << part.local_imax << endl;
		cout<< "Local jmax: " << part.local_jmax << endl;
		cout.flush();
	}
	for (int i = 0; i < part.local_imax; i++)
		for (int j = 0; j < part.local_jmax; j++)
		{
			double x = (i+part.istart) * dx, y = (j+part.jstart) * dy;
			double angle = atan2(y - origin_y, x - origin_x);			//Note that atan2 is not a square, but arctan taking in the x and y components separately

			if (angle > 0.0 && angle < 0.5)
				C1[i * (part.local_jmax) + j] = 0.8;
			else
				C1[i * (part.local_jmax) + j] = q * (f + 1) / (f - 1);

			C2[i * (part.local_jmax) + j] = q * (f + 1) / (f - 1) + angle / (8 * M_PI * f);
		}
}

// Function to clear all files in a folder
void clear_folder(const string& folder_path) {
	for (const auto& entry : __fs::filesystem::directory_iterator(folder_path)) {
		__fs::filesystem::remove(entry.path());
	}
}

// Modified slightly to take in the partition struct, thus the information on local imax and jmax
void grid_to_file(int out, int id, GridPartition part)
{
    // Write the output for a single time step to file
    stringstream fname1, fname2;
    fstream f1, f2;

	string folder_path = "./out/";

    fname1 << "./out/output_C1_" << out << "_" << id << ".dat";
    f1.open(fname1.str().c_str(), ios_base::out);
    fname2 << "./out/output_C2_" << out << "_" << id << ".dat";
    f2.open(fname2.str().c_str(), ios_base::out);

    // Check if files are opened successfully
    if (!f1.is_open() || !f2.is_open()) {
        cout << "Error opening files." << endl;
        return; // Exit the function if files do not open
    }

    // Calculate the local dimensions from the partition
    int local_size = part.local_imax * part.local_jmax;

    // Loop over local grid dimensions
    for (int i = 0; i < part.local_imax; i++) {
        for (int j = 0; j < part.local_jmax; j++) {
            int idx = i * part.local_jmax + j; // Calculate the index in the 1-D array
            // Output local grid values to files
            f1 << C1[idx] << "\t";
            f2 << C2[idx] << "\t";
        }
        f1 << endl;
        f2 << endl;
    }
    f1.close();
    f2.close();
}


int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

    int id, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

	int out_cnt = 0, it = 0;

	//On the root process, clear the output folder.
	if (id == 0) {
		string folder_path = "./out/";
		clear_folder(folder_path);
		string folder_path_combined = "./out_combined/";
		clear_folder(folder_path_combined);
	}


	// Set up the partitioning of the grid
	GridPartition partition = get_partition(imax, jmax, p, id);
	vector<GridPartition> partitions;

    // Create MPI datatype for GridPartition
    MPI_Datatype grid_partition_type = create_grid_partition_type();

    partitions.resize(p);  // Resize to hold all partitions on root

    // Gather all GridPartition structures at root
    MPI_Allgather(&partition, 1, grid_partition_type, partitions.data(), 1, grid_partition_type, MPI_COMM_WORLD);


	// The flag to determine if the domain is periodic
	bool is_periodic = true;
	// Find neighbors for each partition
	find_neighbors(id, partitions, partition, is_periodic);

	if (id == 1) {
		cout << "Partition: " << id << endl;
		cout << "North Neighbor: " << partition.north << endl;
		cout.flush();
		cout << "South Neighbor: " << partition.south << endl;
		cout.flush();
		cout << "East Neighbor: " << partition.east << endl;
		cout.flush();
		cout << "West Neighbor: " << partition.west << endl;
		cout.flush();
	}

	calc_constants();
	
	setup_data_profile(partition,id);

	grid_to_file(out_cnt,id, partition);

	out_cnt++;
	t_out += dt_out;

	// Prepare send and receive buffers
	double* send_buffer_north = new double[partition.local_jmax*2];
	double* send_buffer_south = new double[partition.local_jmax*2];
	double* send_buffer_east = new double[partition.local_imax*2];
	double* send_buffer_west = new double[partition.local_imax*2];

	double* recv_buffer_north = new double[partition.local_jmax*2];
	double* recv_buffer_south = new double[partition.local_jmax*2];
	double* recv_buffer_east = new double[partition.local_imax*2];
	double* recv_buffer_west = new double[partition.local_imax*2];

	// Initialize the buffers to zero
	for (int i = 0; i < partition.local_jmax*2; i++) {
		send_buffer_north[i] = 0.0;
		send_buffer_south[i] = 0.0;
		recv_buffer_north[i] = 0.0;
		recv_buffer_south[i] = 0.0;
	}
	
	// The main while loop
	while (t < t_max)
	{	

		do_iteration(partition, send_buffer_north, send_buffer_south, send_buffer_east, send_buffer_west, recv_buffer_north, recv_buffer_south, recv_buffer_east, recv_buffer_west);

		//Note that I am outputing at a fixed time interval rather than after a fixed number of time steps.
		//This means that the output time interval will be independent of the time step (and thus the resolution)
		if (t_out <= t)
		{
			cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
			grid_to_file(out_cnt,id, partition);
			out_cnt++;
			t_out += dt_out;
		}

		it++;
	}

	// Assemble all timesteps
    int timesteps_per_process = out_cnt / p;
    int start_timestep = id * timesteps_per_process;
    int end_timestep = (id + 1) * timesteps_per_process;

    // Handle the last process case where total_timesteps may not divide evenly
    if (id == p - 1) {
        end_timestep = out_cnt;
    }

    process_timesteps(start_timestep, end_timestep, imax, jmax, partitions, p, id);

	// Clean up the MPI type
    MPI_Type_free(&grid_partition_type);

	// Clean up array
	delete[] C1;
	delete[] C1_old;
	delete[] C2;
	delete[] C2_old;
	delete[] send_buffer_north;
	delete[] send_buffer_south;
	delete[] send_buffer_east;
	delete[] send_buffer_west;
	delete[] recv_buffer_north;
	delete[] recv_buffer_south;
	delete[] recv_buffer_east;
	delete[] recv_buffer_west;

	MPI_Finalize();
	return 0;
}