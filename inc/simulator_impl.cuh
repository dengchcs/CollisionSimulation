#ifndef SIMULATOR_IMPL_CUH
#define SIMULATOR_IMPL_CUH

#include "sphere.hpp"

// kernel使用的模拟配置, 目前限制了原型的数量. 将来可以扩展成支持运行时配置原型数量
struct sim_params {
    int num_spheres;
    float max_radius;
    float radiuses[sphere_proto_num];
    float masses[sphere_proto_num];
    float spring[sphere_proto_num];
    float damping[sphere_proto_num];
    float shear[sphere_proto_num];
    float cell_len;
    float bnd_friction;
    gvec3_t gravity;
};

/**
 * @brief 将全局参数设置拷贝到GPU中
 */
void setup_params(sim_params *params);

/**
 * @brief GPU上进行碰撞检测并更新球体状态
 */
void update_kern(float elapse, sphere *spheres, size_t *hashes, size_t *indices, size_t *cell_start,
                 size_t *cell_end, int num_spheres);

#endif  // SIMULATOR_IMPL_CUH
