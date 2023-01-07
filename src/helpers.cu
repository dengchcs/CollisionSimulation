#include <iostream>

#include "cuda_runtime_api.h"
#include "helpers.cuh"

void check(cudaError_t error, const char *name) {
    if (error != cudaSuccess) {
        std::cerr << name << " " << cudaGetErrorString(error) << '\n';
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
