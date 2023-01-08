#include "simulator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "glm/gtx/hash.hpp"  // for std::hash<gvec3_t>
#include "simulator_impl.cuh"
#include "sphere.hpp"
#include "toml.hpp"

simulator::simulator(const sphere_proto_arr_t &protos, const char *config_path) {
    sphere_protos_ = protos;
    init_sim_params(config_path);
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

    std::random_device rd;
    std::default_random_engine re(rd());
    auto random_float = [&](float max_abs) -> float {
        std::uniform_real_distribution<float> dist(-max_abs, max_abs);
        return dist(re);
    };
    std::vector<int> proto_types;
    proto_types.reserve(sim_params_.num_spheres);
    for (int i = 0; i < sphere_proto_num; i++) {
        proto_types.insert(proto_types.end(), sphere_protos_[i].num, i);
    }
    std::shuffle(proto_types.begin(), proto_types.end(), re);

    const int num_spheres = sim_params_.num_spheres;
    const int nx = std::floor(std::pow(num_spheres, 1.0F / 3.0F));
    const int nz = nx;
    std::unordered_set<gvec3_t> added_pos;
    for (int i = 0; i < num_spheres; i++) {
        const int index_x = i % nx;
        const int index_z = (i - index_x) / nx % nz;
        const int index_y = (i - index_x - nx * index_z) / (nx * nz);
        spheres_[i].pos = {-1.F + cell_len / 2.F + cell_len * index_x,
                           1.F - (cell_len / 2.F + cell_len * index_y),
                           -1.F + cell_len / 2.F + cell_len * index_z};
        added_pos.insert(spheres_[i].pos);
        spheres_[i].veloc = {
            random_float(0.005F * cell_len),
            random_float(0.001F * cell_len),
            random_float(0.005F * cell_len),
        };
        spheres_[i].accel = {0.F, 0.F, 0.F};
        spheres_[i].type = proto_types.back();
        proto_types.pop_back();
    }
    if (added_pos.size() != num_spheres) {
        std::cerr << "sphere at the same pos!!!\n";
    }
}

void simulator::init_sim_params(const char *config_path) {
    sim_params_.num_spheres = 0;
    sim_params_.max_radius = 0;
    for (int i = 0; i < sphere_proto_num; i++) {
        const auto proto = sphere_protos_[i];
        sim_params_.num_spheres += proto.num;
        if (proto.num > 0) {
            sim_params_.max_radius = std::max(sim_params_.max_radius, proto.radius);
        }
        sim_params_.radiuses[i] = proto.radius;
        sim_params_.masses[i] = proto.mass;
        sim_params_.spring[i] = proto.spring;
        sim_params_.damping[i] = proto.damping;
        sim_params_.shear[i] = proto.shear;
    }
    sim_params_.cell_len = 2 * sim_params_.max_radius;

    const auto physics = toml::parse_file(config_path)["physics"];
    sim_params_.bnd_friction = physics["bnd_friction"].value_or(0.1F);
    const auto gravity = physics["gravity"];
    sim_params_.gravity = {gravity[0].value_or(0.0F), gravity[1].value_or(-0.5F),
                           gravity[2].value_or(0.0F)};

    setup_params(&sim_params_);
}

void simulator::update(float elapse) {
    update_kern(elapse, spheres_, hashes_, indices_, cell_start_, cell_end_,
                sim_params_.num_spheres);
};
