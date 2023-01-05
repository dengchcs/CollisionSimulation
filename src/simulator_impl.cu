#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdio>

#include "common.hpp"
#include "glm/geometric.hpp"
#include "helpers.cuh"
#include "simulator_impl.cuh"
#include "sphere.hpp"

__constant__ sim_params params;

#define THREAD_INDEX ((blockIdx.x * blockDim.x) + threadIdx.x)

__device__ auto calc_cell_pos(gvec3_t pos) -> gvec3i_t {
    return {
        std::floor(pos.x / params.cell_len),
        std::floor(pos.y / params.cell_len),
        std::floor(pos.z / params.cell_len),
    };
}

void setup_params(sim_params *params_in) {
    check(cudaMemcpyToSymbol(params, params_in, sizeof(sim_params)), "setup_params()");
}

/**
 * @brief 计算两球相撞时球1的受力
 */
__device__ auto collide_two_dev(sphere sph1, sphere sph2) -> gvec3_t {
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

__global__ void calc_cell_hash(size_t *hashes, size_t *indices, const sphere *spheres) {
    const auto index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    const gvec3_t current_pos = spheres[index].pos;
    const auto cell_pos = calc_cell_pos(current_pos);
    const auto hash = index;  // TODO:
    hashes[index] = hash;
    indices[index] = index;
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
            veloc[axis] = -std::abs(veloc[axis]);
        } else if (pos[axis] < -1.F + radius) {
            pos[axis] = -1.F + radius;
            veloc[axis] = std::abs(veloc[axis]);
        }
    };
    clamp_axis(0);
    clamp_axis(1);
    clamp_axis(2);

    spheres[index].pos = pos;
    spheres[index].veloc = veloc;
}

void update_kern(float elapse, sphere* spheres, size_t *hashes,
                 size_t *indices, int num_spheres) {
    // 不能在host中读取constant内存, 所以需要指定参数num_spheres
    const int num_threads = std::min(256, num_spheres);
    const int num_blocks = (num_spheres + num_threads - 1) / num_threads;
    calc_cell_hash<<<num_blocks, num_threads>>>(hashes, indices, spheres);
    integrate<<<num_blocks, num_threads>>>(elapse, spheres);
    check(cudaDeviceSynchronize(), "sync after update");
}
