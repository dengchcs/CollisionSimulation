﻿#ifndef SIMULATOR_IMPL_CUH
#define SIMULATOR_IMPL_CUH

#include <cstddef>

#include "cuda_runtime.h"
#include "sphere.hpp"

struct sim_params {
    int num_spheres;
    float max_radius;
    float radiuses[sphere_proto_num];
    float masses[sphere_proto_num];
    // float damping[sphere_proto_num][sphere_proto_num];
    // float restitution[sphere_proto_num][sphere_proto_num];
    // TODO: 改成每个小球各不相同的情况
    float spring;
    float damping;
    float shear;
    float cell_len;
};

/**
 * @brief 将全局参数设置拷贝到GPU中
 */
void setup_params(sim_params *params);

/**
 * @brief GPU上进行碰撞检测并更新球体状态
 *
 * @param elapse
 * @param pos
 * @param veloc
 * @param accel
 * @param type
 * @param hashes
 * @param indices
 */
void update_kern(float elapse, sphere *spheres, size_t *hashes, size_t *indices, size_t *cell_start,
                 size_t *cell_end, int num_spheres);

#endif  // SIMULATOR_IMPL_CUH
