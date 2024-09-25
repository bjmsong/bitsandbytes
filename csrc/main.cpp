#include <iostream>
#include <random>
#include <cstdio>
#include <ops.cuh>
#include <cuda_runtime.h>

std::default_random_engine random_engine(0);
std::uniform_real_distribution<float> uniform_dist(-256, 256);

void matrix_init(half* a, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = __float2half(uniform_dist(random_engine));
        }
    }
}

void matrix_init(float* a, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = uniform_dist(random_engine);
        }
    }
}

void matrix_init(float* a, int M, int N, float value){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            a[i*N + j] = value;
        }
    }
}

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


int main(int argc, char **argv) {

    int M = 256;
    int N = 512;
    size_t bytes_a = M * N * sizeof(half);
    half* h_a = (half*)malloc(bytes_a);
    matrix_init(h_a, M, N);
    half *d_a;
    checkCuda(cudaMalloc(&d_a, bytes_a));
    checkCuda(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));

    size_t bytes_rowStatus = M * sizeof(float);
    float* h_rowStatus = (float*)malloc(bytes_rowStatus);
    matrix_init(h_rowStatus, M, 1, -50000.0);
    float *d_rowStatus;
    checkCuda(cudaMalloc(&d_rowStatus, bytes_rowStatus));
    checkCuda(cudaMemcpy(d_rowStatus, h_rowStatus, bytes_rowStatus, cudaMemcpyHostToDevice));

    size_t bytes_colStats = N * sizeof(float);
    float* h_colStats = (float*)malloc(bytes_colStats);
    matrix_init(h_colStats, 1, M, -50000.0);
    float *d_colStats;
    checkCuda(cudaMalloc(&d_colStats, bytes_colStats));
    checkCuda(cudaMemcpy(d_colStats, h_colStats, bytes_colStats, cudaMemcpyHostToDevice));
    
    float nnz_threshold = 0.0f;

    getColRowStats(d_a, d_rowStatus, d_colStats, nullptr, nnz_threshold, M, N);
}