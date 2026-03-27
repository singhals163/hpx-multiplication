#include <hpx/hpx_main.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// A simple 1D-backed matrix structure for contiguous memory access
struct Matrix {
    size_t rows, cols;
    std::vector<double> data;
    
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    double& operator()(size_t i, size_t j) { 
        return data[i * cols + j]; 
    }
    const double& operator()(size_t i, size_t j) const { 
        return data[i * cols + j]; 
    }
};

// Parallelize the computation over the rows of the resulting matrix C
void multiply_parallel(const Matrix& A, const Matrix& B, Matrix& C) {
    hpx::experimental::for_loop(hpx::execution::par, 0, A.rows, [&](size_t i) {
        for (size_t j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    });
}

int main() {
    size_t size = 512; // 512x512 matrix
    Matrix A(size, size);
    Matrix B(size, size);
    Matrix C(size, size);

    // Initialize matrices in parallel
    hpx::experimental::for_loop(hpx::execution::par, 0, size * size, [&](size_t i) {
        A.data[i] = 1.0;
        B.data[i] = 2.0;
    });


    auto start = std::chrono::high_resolution_clock::now();
    multiply_parallel(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "Matrix Multiplication (" << size << "x" << size << ") completed in " 
              << diff.count() << " seconds.\n";
              
    // Verification: C(0,0) should be size * (1.0 * 2.0) = 1024.0
    std::cout << "Top-left element of C: " << C(0, 0) << "\n"; 

    return 0;
}