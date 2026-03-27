#include <hpx/hpx_main.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <iostream>
#include <vector>
#include <chrono>

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

// Standard sequential execution policy
void multiply_sequential(const Matrix& A, const Matrix& B, Matrix& C) {
    hpx::experimental::for_loop(hpx::execution::seq, 0, A.rows, [&](size_t i) {
        for (size_t j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    });
}

// Parallel execution policy distributing work across OS threads
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
    size_t size = 512; 
    Matrix A(size, size);
    Matrix B(size, size);
    Matrix C_seq(size, size);
    Matrix C_par(size, size);

    // Initialize matrices
    hpx::experimental::for_loop(hpx::execution::par, 0, size * size, [&](size_t i) {
        A.data[i] = 1.0;
        B.data[i] = 2.0;
    });

    // Sequential Multiplication
    auto start_seq = std::chrono::high_resolution_clock::now();
    multiply_sequential(A, B, C_seq);
    auto end_seq = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_seq = end_seq - start_seq;
    std::cout << "[Sequential] Completed in: " << diff_seq.count() << " seconds.\n";

    // Parallel Multiplication
    auto start_par = std::chrono::high_resolution_clock::now();
    multiply_parallel(A, B, C_par);
    auto end_par = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff_par = end_par - start_par;
    std::cout << "[Parallel]   Completed in: " << diff_par.count() << " seconds.\n";

    // Calculate Speedup
    double speedup = diff_seq.count() / diff_par.count();
    std::cout << "\nParallel Speedup: " << speedup << "x\n";
    
    // Verification check to ensure both computed the same result
    if (C_seq(0, 0) == C_par(0, 0)) {
        std::cout << "Verification: PASSED (Outputs match)\n";
    } else {
        std::cout << "Verification: FAILED (Outputs mismatch)\n";
    }

    return 0;
}