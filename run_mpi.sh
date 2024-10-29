#!/bin/bash

# Create a build directory and navigate into it
mkdir -p build
cd build

# Configure the project with CMake
cmake ..

# Build the project
make

# Run the MPI program with a specific number of processes
mpirun -np 4 ./cw

cd ..

cd python_code

python post_process.py
# ./serial