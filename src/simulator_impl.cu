﻿#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdio.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

#include "common.hpp"
#include "glm/geometric.hpp"
#include "helpers.cuh"
#include "simulator_impl.cuh"
#include "sphere.hpp"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

__constant__ sim_params params;

#define THREAD_INDEX ((blockIdx.x * blockDim.x) + threadIdx.x)

__device__ auto calc_cell_pos(gvec3_t pos) -> gvec3i_t {
    return {
        std::floor((pos.x + 1.F) / params.cell_len),
        std::floor((pos.y + 1.F) / params.cell_len),
        std::floor((pos.z + 1.F) / params.cell_len),
    };
}

void setup_params(sim_params *params_in) {
    check(cudaMemcpyToSymbol(params, params_in, sizeof(sim_params)), "setup_params()");
}

__device__ void print_vec3(const char *leader, gvec3_t vec) {
    printf("%s (%f, %f, %f)", leader, vec.x, vec.y, vec.z);
}

__device__ auto hash_func(gvec3i_t pos) -> size_t { return (pos.x << 16) | (pos.y << 8) | (pos.z); }

__global__ void calc_cell_hash(size_t *hashes, size_t *indices, const sphere *spheres) {
    const auto index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    const gvec3_t current_pos = spheres[index].pos;
    const auto cell_pos = calc_cell_pos(current_pos);
    const auto hash = hash_func(cell_pos);  // TODO:
    hashes[index] = hash;
    indices[index] = index;
}

__global__ void fill_cells(size_t *cell_start, size_t *cell_end, const size_t *hashes) {
    const size_t index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    const size_t hash = hashes[index];
    if (index == 0) {
        cell_start[index] = index;
    } else if (hash != hashes[index - 1]) {
        cell_start[hash] = index;
        cell_end[hashes[index - 1]] = index;
    }
    if (index == params.num_spheres - 1) {
        cell_end[hash] = index + 1;
    }
}

/**
 * @brief 计算两球相撞时球1的受力
 */
__device__ auto collide_two(sphere sph1, sphere sph2) -> gvec3_t {
    const float radius1 = params.radiuses[sph1.type];
    const float radius2 = params.radiuses[sph2.type];
    const float radius_sum = radius1 + radius2;
    const auto rel_pos = sph2.pos - sph1.pos;
    const float dist = glm::length(rel_pos);
    gvec3_t force{0.F};
    if (dist >= radius_sum) {
        return force;
    }

    gvec3_t normal = glm::normalize(rel_pos);
    gvec3_t rel_veloc = sph2.veloc - sph1.veloc;
    gvec3_t tan_veloc = rel_veloc - glm::dot(rel_veloc, normal) * normal;

    force -= params.spring * (radius_sum - dist) * normal;
    force += params.damping * rel_veloc;
    force += params.shear * tan_veloc;
    return force;
}

/**
 * @brief 计算每个球的受力, 更新其加速度
 */
__global__ void collide_all(sphere *spheres, const size_t *indices, const size_t *cell_start,
                            const size_t *cell_end) {
    const size_t thread_idx = THREAD_INDEX;
    if (thread_idx >= params.num_spheres) {
        return;
    }
    const size_t sphere_idx = indices[thread_idx];
    const auto target = spheres[sphere_idx];
    gvec3_t force{0.F};

    int cnt = 0;
    const auto cell_pos = calc_cell_pos(target.pos);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                const auto neighbor = cell_pos + gvec3i_t{x, y, z};
                const size_t hash_ngh = hash_func(neighbor);
                const size_t idx_start = cell_start[hash_ngh];
                if (idx_start != 0xFFFFFFFFFFFFFFFF) {
                    const size_t idx_end = cell_end[hash_ngh];
                    for (size_t i = idx_start; i < idx_end; i++) {
                        const auto idx = indices[i];
                        if (idx != sphere_idx) {
                            force += collide_two(target, spheres[idx]);
                        }
                    }
                }
            }
        }
    }

    const float mass = params.masses[target.type];
    spheres[sphere_idx].accel = force / mass;
}

__global__ void integrate(float elapse, sphere *spheres) {
    const auto index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    gvec3_t pos = spheres[index].pos;
    gvec3_t veloc = spheres[index].veloc;
    const gvec3_t accel = spheres[index].accel;
    const auto type = spheres[index].type;
    const auto radius = params.radiuses[type];

    veloc += accel * elapse;
    veloc += gvec3_t{0.F, -1.F, 0.F} * elapse;
    pos += veloc * elapse;

    auto clamp_axis = [&](int axis) {
        if (pos[axis] > 1.F - radius) {
            pos[axis] = 1.F - radius;
            veloc[axis] = -std::abs(veloc[axis]) * 0.5F;
        } else if (pos[axis] < -1.F + radius) {
            pos[axis] = -1.F + radius;
            veloc[axis] = std::abs(veloc[axis]) * 0.5F;
        }
    };
    clamp_axis(0);
    clamp_axis(1);
    clamp_axis(2);

    spheres[index].pos = pos;
    spheres[index].veloc = veloc;
}

void update_kern(float elapse, sphere *spheres, size_t *hashes, size_t *indices, size_t *cell_start,
                 size_t *cell_end, int num_spheres) {
    // 不能在host中读取constant内存, 所以需要指定参数num_spheres
    const int num_threads = std::min(256, num_spheres);
    const int num_blocks = (num_spheres + num_threads - 1) / num_threads;

    calc_cell_hash<<<num_blocks, num_threads>>>(hashes, indices, spheres);
    thrust::sort_by_key(thrust::device_ptr<size_t>(hashes),
                        thrust::device_ptr<size_t>(hashes + num_spheres),
                        thrust::device_ptr<size_t>(indices));
    check(cudaMemset(cell_start, 0XFF, (1 << 24) * sizeof(size_t)), "memset cell_start");
    fill_cells<<<num_blocks, num_threads>>>(cell_start, cell_end, hashes);
    collide_all<<<num_blocks, num_threads>>>(spheres, indices, cell_start, cell_end);
    integrate<<<num_blocks, num_threads>>>(elapse, spheres);

    check(cudaDeviceSynchronize(), "sync after update");
}
