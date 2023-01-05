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

__device__ auto vec3_ith(const float *pos_vec, size_t index) -> gvec3_t {
    const auto index_3 = index * 3;
    return {pos_vec[index_3], pos_vec[index_3 + 1], pos_vec[index_3 + 2]};
}

__device__ void set_vec3_ith(float *vec, size_t index, float x, float y, float z) {
    const auto index_3 = index * 3;
    vec[index_3 + 0] = x;
    vec[index_3 + 1] = y;
    vec[index_3 + 2] = z;
}

__global__ void test() {
    auto force = gvec4_t{1.0F};
    auto len = glm::length(force);
    printf("damping = %f\n", params.damping);
}

void setup_params(sim_params *params_in) {
    check(cudaMemcpyToSymbol(params, params_in, sizeof(sim_params)), "setup_params()");
}

/**
 * @brief 计算两球相撞时球1的受力
 */
__device__ auto collide_two_dev(sphere sph1, sphere sph2) -> gvec3_t {
    const float radius1 = sph1.proto.radius;
    const float radius2 = sph2.proto.radius;
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

__global__ void calc_cell_hash(size_t *hashes, size_t *indices, const float *pos) {
    const auto index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    const gvec3_t current_pos = vec3_ith(pos, index);
    const auto cell_pos = calc_cell_pos(current_pos);
    const auto hash = index;  // TODO:
    hashes[index] = hash;
    indices[index] = index;
}

__global__ void integrate(float elapse, float *pos_vec, float *veloc_vec, const float *accel_vec,
                          const size_t *type_vec) {
    const auto index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    gvec3_t pos = vec3_ith(pos_vec, index);
    gvec3_t veloc = vec3_ith(veloc_vec, index);
    const gvec3_t accel = vec3_ith(accel_vec, index);
    const auto type = type_vec[index];
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

    set_vec3_ith(pos_vec, index, pos.x, pos.y, pos.z);
    set_vec3_ith(veloc_vec, index, veloc.x, veloc.y, veloc.z);
}

void update_kern(float elapse, float *pos, float *veloc, float *accel, size_t *type, size_t *hashes,
                 size_t *indices, int num_spheres) {
    // 不能在host中读取constant内存, 所以需要指定参数num_spheres
    const int num_threads = std::min(256, num_spheres);
    const int num_blocks = (num_spheres + num_threads - 1) / num_threads;
    calc_cell_hash<<<num_blocks, num_threads>>>(hashes, indices, pos);
    integrate<<<num_blocks, num_threads>>>(elapse, pos, veloc, accel, type);
    check(cudaDeviceSynchronize(), "sync after update");
}
