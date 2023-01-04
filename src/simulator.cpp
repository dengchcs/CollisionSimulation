#include "simulator.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "simulator_impl.cuh"
#include "sphere.hpp"

simulator::simulator() {
    for (int i = 0; i < sphere_proto_num; i++) {
        sphere_num_ += sphere_protos[i].num;
        max_radius_ = std::max(max_radius_, sphere_protos[i].radius);
    }
    init_memory();
    set_initial_state();
}

simulator::~simulator() { free_memory(); }

void simulator::init_memory() {
    sphere_pos_ = new float[3 * sphere_num_];
    sphere_velo_ = new float[3 * sphere_num_];
    sphere_accel_ = new float[3 * sphere_num_];
    sphere_type_ = new int[sphere_num_];
}

void simulator::free_memory() {
    delete[] sphere_pos_;
    delete[] sphere_velo_;
    delete[] sphere_accel_;
    delete[] sphere_type_;
}

void simulator::set_initial_state() {
    const float cell_len = max_radius_ * 2.F;
    const int num_per_axis = std::floor(2.0F / cell_len);
    const int cell_num = num_per_axis * num_per_axis * num_per_axis;
    if (cell_num < sphere_num_) {
        std::cerr << "too many objects to fit in the space\n";
        exit(1);
    }
    for (int i = 0; i < sphere_num_; i++) {
        // i = per_axis^2 * index_y + per_axis * index_z + index_x
        const int index_x = i % num_per_axis;
        const int index_z = (i - index_x) / num_per_axis % num_per_axis;
        const int index_y = i / (num_per_axis * num_per_axis);
        const int pos_start = i * 3;
        sphere_pos_[pos_start + 0] = -1.F + cell_len / 2.F + cell_len * index_x;
        sphere_pos_[pos_start + 1] = 1.F - (cell_len / 2.F + cell_len * index_y);
        sphere_pos_[pos_start + 2] = -1.F + cell_len / 2.F + cell_len * index_z;
        sphere_type_[i] = rand() % sphere_proto_num;
        // TODO:
        sphere_velo_[pos_start + 0] = 0;
        sphere_velo_[pos_start + 1] = 0;
        sphere_velo_[pos_start + 2] = 0;
        sphere_accel_[pos_start + 0] = 0;
        sphere_accel_[pos_start + 1] = 0;
        sphere_accel_[pos_start + 2] = 0;
    }
}

void simulator::update(float elapse){
    // TODO:
};
