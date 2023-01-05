#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <cstddef>

#include "simulator_impl.cuh"

class simulator {
    float *pos_ = nullptr;    // 每个球的XYZ坐标
    float *veloc_ = nullptr;  // 每个球XYZ方向的速度
    float *accel_ = nullptr;  // 每个球XYZ方向的加速度
    size_t *type_ = nullptr;  // 每个球的类型(在protos中的下标)
    size_t *hashes_ = nullptr;
    size_t *indices_ = nullptr;

    sim_params sim_params_;

    void set_initial_state();
    void init_memory();
    void free_memory();

    void init_sim_params();

public:
    simulator();

    ~simulator();

    [[nodiscard]] auto sphere_pos() const -> const float * { return pos_; }
    [[nodiscard]] auto sphere_type() const -> const size_t * { return type_; }
    [[nodiscard]] auto sphere_num() const -> int { return sim_params_.num_spheres; }

    void update(float elapse);
};

#endif  // SIMULATOR_HPP