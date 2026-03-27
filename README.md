
# HPX Matrix Multiplication: Sequential vs. Parallel Performance

This repository contains a C++ implementation of dense matrix multiplication utilizing the [HPX](https://github.com/STEllAR-GROUP/hpx) runtime system. It serves as a practical demonstration of task-based parallelism, explicitly comparing the performance of single-threaded execution against multi-core distributed execution.

## Repository Structure

* `main.cpp`: The core application. It defines the contiguous matrix data structure, initializes the matrices, executes both sequential and parallel multiplications, and calculates the resulting speedup.
* `CMakeLists.txt`: The build configuration file required to locate the local HPX installation, enforce C++17 standards, and link the necessary HPX libraries (including `HPX::wrap_main`).

## Build and Run Instructions

**Prerequisites:** You must have HPX built and installed on your system.

1. Create a build directory and configure the project using CMake:
   ```bash
   mkdir build && cd build
   cmake -DHPX_DIR=/path/to/your/hpx/installation/lib/cmake/HPX ..
   ```
