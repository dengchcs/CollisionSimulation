#include "simulator.hpp"

#include <stddef.h>
#include <stdlib.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "simulator_impl.cuh"
#include "sphere.hpp"

simulator::simulator() {
    init_sim_params();
    init_memory();
    set_initial_state();
}

simulator::~simulator() { free_memory(); }

void simulator::init_memory() {
    const auto nbytes_1u = sim_params_.num_spheres * sizeof(size_t);
    cudaMallocManaged(&hashes_, nbytes_1u);
    cudaMallocManaged(&indices_, nbytes_1u);
    cudaMallocManaged(&cell_start_, (1 << 24) * sizeof(size_t));
    cudaMallocManaged(&cell_end_, (1 << 24) * sizeof(size_t));

    cudaMallocManaged(&spheres_, sim_params_.num_spheres * sizeof(sphere));
}

void simulator::free_memory() {
    cudaFree(spheres_);
    cudaFree(hashes_);
    cudaFree(indices_);
    cudaFree(cell_start_);
    cudaFree(cell_end_);
}

void simulator::set_initial_state() {
    const float cell_len = sim_params_.cell_len;
    const int num_per_axis = std::floor(2.0F / cell_len);
    const int cell_num = num_per_axis * num_per_axis * num_per_axis;
    if (cell_num < sim_params_.num_spheres) {
        std::cerr << "too many objects to fit in the space\n";
        exit(1);
    }
    auto random_float = [](float max_abs) -> float {
        std::uniform_real_distribution<float> dist(-max_abs, max_abs);
        static std::random_device rd;
        static std::default_random_engine re(rd());
        return dist(re);
    };
    for (int i = 0; i < sim_params_.num_spheres; i++) {
        // i = per_axis^2 * index_y + per_axis * index_z + index_x
        const int index_x = i % num_per_axis;
        const int index_z = (i - index_x) / num_per_axis % num_per_axis;
        const int index_y = i / (num_per_axis * num_per_axis);
        const int pos_start = i * 3;
        spheres_[i].pos = {-1.F + cell_len / 2.F + cell_len * index_x,
                           1.F - (cell_len / 2.F + cell_len * index_y),
                           -1.F + cell_len / 2.F + cell_len * index_z};
        spheres_[i].veloc = {
            random_float(0.5F * cell_len),
            random_float(0.5F * cell_len),
            random_float(0.5F * cell_len),
        };
        spheres_[i].accel = {0.F, 0.F, 0.F};
        spheres_[i].type = rand() % sphere_proto_num;
    }
}

void simulator::init_sim_params() {
    sim_params_.num_spheres = 0;
    sim_params_.max_radius = 0;
    for (int i = 0; i < sphere_proto_num; i++) {
        const auto proto = sphere_protos[i];
        sim_params_.num_spheres += proto.num;
        sim_params_.max_radius = std::max(sim_params_.max_radius, proto.radius);
        sim_params_.radiuses[i] = proto.radius;
        sim_params_.masses[i] = proto.mass;
    }
    sim_params_.spring = 1000.F;
    sim_params_.damping = 0.02F;
    sim_params_.shear = 0.1F;
    sim_params_.cell_len = 2 * sim_params_.max_radius;
    // TODO: 设置更多参数
    setup_params(&sim_params_);
}

void simulator::update(float elapse) {
    update_kern(elapse, spheres_, hashes_, indices_, cell_start_, cell_end_,
                sim_params_.num_spheres);
};
