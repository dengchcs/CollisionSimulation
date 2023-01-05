#include "simulator.hpp"

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
    test();
}

simulator::~simulator() { free_memory(); }

void simulator::init_memory() {
    const auto nbytes_3f = 3LL * sim_params_.num_spheres * sizeof(float);
    cudaMallocManaged(&pos_, nbytes_3f);
    cudaMallocManaged(&veloc_, nbytes_3f);
    cudaMallocManaged(&accel_, nbytes_3f);
    const auto nbytes_1s = sim_params_.num_spheres * sizeof(size_t);
    cudaMallocManaged(&type_, nbytes_1s);
    cudaMallocManaged(&hashes_, nbytes_1s);
    cudaMallocManaged(&indices_, nbytes_1s);
}

void simulator::free_memory() {
    cudaFree(pos_);
    cudaFree(veloc_);
    cudaFree(accel_);
    cudaFree(type_);
    cudaFree(hashes_);
    cudaFree(indices_);
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
        pos_[pos_start + 0] = -1.F + cell_len / 2.F + cell_len * index_x;
        pos_[pos_start + 1] = 1.F - (cell_len / 2.F + cell_len * index_y);
        pos_[pos_start + 2] = -1.F + cell_len / 2.F + cell_len * index_z;
        type_[i] = rand() % sphere_proto_num;
        // TODO:
        veloc_[pos_start + 0] = random_float(0.5F * cell_len);
        veloc_[pos_start + 1] = random_float(0.5F * cell_len);
        veloc_[pos_start + 2] = random_float(0.5F * cell_len);
        accel_[pos_start + 0] = 0;
        accel_[pos_start + 1] = 0;
        accel_[pos_start + 2] = 0;
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
    sim_params_.spring = 0.5F;
    sim_params_.damping = 0.02F;
    sim_params_.shear = 0.1F;
    sim_params_.cell_len = 2 * sim_params_.max_radius;
    // TODO: 设置更多参数
    setup_params(&sim_params_);
}

void simulator::update(float elapse) {
    update_kern(elapse, pos_, veloc_, accel_, type_, hashes_, indices_, sim_params_.num_spheres);
};
