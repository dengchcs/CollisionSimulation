#include <cmath>
#include <cstdio>

#include "common.hpp"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "glm/geometric.hpp"  // glm::length()/normalize()/...
#include "simulator_impl.cuh"
#include "sphere.hpp"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

__constant__ sim_params params;

#define THREAD_INDEX ((blockIdx.x * blockDim.x) + threadIdx.x)

/**
 * @brief 检查cuda执行结果, 出错时报错并退出
 */
void check(cudaError_t error, const char *name) {
    if (error != cudaSuccess) {
        std::cerr << name << " " << cudaGetErrorString(error) << '\n';
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void setup_params(sim_params *params_in) {
    check(cudaMemcpyToSymbol(params, params_in, sizeof(sim_params)), "setup_params()");
}

/**
 * @brief 将世界坐标转换成网格坐标, 世界(-1,-1,-1) -> 网格(0, 0, 0)
 */
__device__ auto calc_cell_pos(gvec3_t pos) -> gvec3i_t {
    return {
        std::floor((pos.x + 1.F) / params.cell_len),
        std::floor((pos.y + 1.F) / params.cell_len),
        std::floor((pos.z + 1.F) / params.cell_len),
    };
}

/**
 * @brief 计算网格哈希: 将所有bits拼接起来
 * @note 目前的实现每条边的网格数不超过1<<8=256个
 */
__device__ auto hash_func(gvec3i_t pos) -> size_t { return (pos.x << 16) | (pos.y << 8) | (pos.z); }

/**
 * @brief 计算所有球体所在网格的哈希值
 */
__global__ void calc_cell_hash(size_t *hashes, size_t *indices, const sphere *spheres) {
    const auto index = THREAD_INDEX;
    if (index >= params.num_spheres) {
        return;
    }
    const gvec3_t sphere_pos = spheres[index].pos;
    const auto cell_pos = calc_cell_pos(sphere_pos);
    const auto hash = hash_func(cell_pos);
    hashes[index] = hash;
    indices[index] = index;
}

/**
 * @brief 记录每个hash对应网格包含球体在indices中的起始&终止位置
 */
__global__ void fill_cells(size_t *cell_start, size_t *cell_end, const size_t *hashes_sorted) {
    // 此处的idx是在排序后的indices数组中的index, 对应的球体spheres[indices[idx]]
    const size_t indices_idx = THREAD_INDEX;
    if (indices_idx >= params.num_spheres) {
        return;
    }
    const size_t hash = hashes_sorted[indices_idx];
    if (indices_idx == 0) {
        cell_start[hash] = indices_idx;
    } else if (hash != hashes_sorted[indices_idx - 1]) {
        cell_start[hash] = indices_idx;
        cell_end[hashes_sorted[indices_idx - 1]] = indices_idx;
    }
    if (indices_idx == params.num_spheres - 1) {
        cell_end[hash] = indices_idx + 1;
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
    /*
    if (dist == 0.0F) {
        printf("error: zero-length vector!!!\n");
    }
    */
    gvec3_t force{0.F};
    if (dist >= radius_sum || dist == 0.0F) {
        return force;
    }

    // 按照文档, normalize()作用在零向量上的结果是未定义的
    // 考虑到计算误差, 这种情况是有可能出现的, 所以在上面规避了单位化零向量
    gvec3_t normal = glm::normalize(rel_pos);
    gvec3_t rel_veloc = sph2.veloc - sph1.veloc;
    gvec3_t tan_veloc = rel_veloc - glm::dot(rel_veloc, normal) * normal;

    force -= params.spring[sph1.type] * (radius_sum - dist) * normal;
    force += params.damping[sph1.type] * rel_veloc;
    force += params.shear[sph1.type] * tan_veloc;
    return force;
}

/**
 * @brief 计算每个球的受力, 更新其加速度
 */
__global__ void collide_all(sphere *spheres, const size_t *indices_sorted, const size_t *cell_start,
                            const size_t *cell_end) {
    const size_t thread_idx = THREAD_INDEX;
    if (thread_idx >= params.num_spheres) {
        return;
    }
    const size_t sphere_idx = indices_sorted[thread_idx];
    const auto target = spheres[sphere_idx];
    gvec3_t force{0.F};

    const auto cell_pos = calc_cell_pos(target.pos);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                const auto neighbor = cell_pos + gvec3i_t{x, y, z};
                if (neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0) {
                    continue;
                }
                const size_t hash_ngh = hash_func(neighbor);
                const size_t idx_start = cell_start[hash_ngh];
                if (idx_start == 0xFFFFFFFFFFFFFFFF) {
                    continue;
                }
                const size_t idx_end = cell_end[hash_ngh];
                for (size_t i = idx_start; i < idx_end; i++) {
                    const auto idx = indices_sorted[i];
                    if (idx != sphere_idx) {
                        force += collide_two(target, spheres[idx]);
                    }
                }
            }
        }
    }

    const float mass = params.masses[target.type];
    const auto accel = force / mass;
    spheres[sphere_idx].accel = accel;
}

/**
 * @brief 更新球体在运动给定时间后的状态
 */
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
    const auto spring = params.spring[type];

    veloc += accel * elapse;
    veloc += params.gravity * elapse;
    pos += veloc * elapse;

    // 处理与边界的碰撞
    auto clamp_axis = [&](int axis) {
        bool bnd_touch = false;
        const float v_axis_old_abs = std::abs(veloc[axis]);
        if (pos[axis] > 1.F - radius) {
            bnd_touch = true;
            pos[axis] = 1.F - radius;
            veloc[axis] = -spring * v_axis_old_abs;
        } else if (pos[axis] < -1.F + radius) {
            bnd_touch = true;
            pos[axis] = -1.F + radius;
            veloc[axis] = spring * v_axis_old_abs;
        }
        if (bnd_touch) {
            const float k = params.bnd_friction * (1 + spring);
            const float v_plane1 = veloc[(axis + 1) % 3];
            const float v_plane2 = veloc[(axis + 2) % 3];
            const float v_plane_old = std::sqrt(v_plane1 * v_plane1 + v_plane2 * v_plane2);
            if (v_plane_old > 0.F) {
                float v_plane_new = v_plane_old - k * v_axis_old_abs;
                if (v_plane_new <= 0.F) {
                    v_plane_new = 0.F;
                }
                veloc[(axis + 1) % 3] = v_plane1 / v_plane_old * v_plane_new;
                veloc[(axis + 2) % 3] = v_plane2 / v_plane_old * v_plane_new;
            }
        }
    };
    clamp_axis(0);
    clamp_axis(1);
    clamp_axis(2);

    spheres[index].pos = pos;
    spheres[index].veloc = veloc;

    auto correct_nan = [](gvec3_t &vec3) {
        if (isnan(vec3.x) || isnan(vec3.y) || isnan(vec3.z)) {
            printf("bad vec3 during integrating: (%f, %f, %f)\n", vec3.x, vec3.y, vec3.z);
            vec3 = {0.0F, 0.0F, 0.0F};
        }
    };
    correct_nan(spheres[index].pos);
    correct_nan(spheres[index].veloc);
    correct_nan(spheres[index].accel);
}

void update_kern(float elapse, sphere *spheres, size_t *hashes, size_t *indices, size_t *cell_start,
                 size_t *cell_end, int num_spheres) {
    // 不能在host中读取constant内存, 所以需要指定参数num_spheres
    const int num_threads = std::min(256, num_spheres);
    const int num_blocks = (num_spheres + num_threads - 1) / num_threads;

    // kenel在GPU上是同步执行的, 所以只需要在所有kernel launch后做一次sync()

    calc_cell_hash<<<num_blocks, num_threads>>>(hashes, indices, spheres);
    // 将所有球体的下标按其所处cell的哈希值排序, 这样同一cell中的球体的下标在排序后的
    // indices数组中是连续存储的, 方便碰撞检测时快速查询所有可能碰撞对象
    thrust::sort_by_key(thrust::device_ptr<size_t>(hashes),
                        thrust::device_ptr<size_t>(hashes + num_spheres),
                        thrust::device_ptr<size_t>(indices));
    check(cudaMemset(cell_start, 0XFF, (1 << 24) * sizeof(size_t)), "memset cell_start");
    fill_cells<<<num_blocks, num_threads>>>(cell_start, cell_end, hashes);

    collide_all<<<num_blocks, num_threads>>>(spheres, indices, cell_start, cell_end);
    integrate<<<num_blocks, num_threads>>>(elapse, spheres);

    check(cudaDeviceSynchronize(), "sync after update");
}
