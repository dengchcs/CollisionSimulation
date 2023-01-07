#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <array>
#include <cstddef>

#include "common.hpp"

/**
 * @brief 球体的"原型", 不包含运动信息
 */
struct sphere_proto {
    float spring;
    float damping;
    float shear;
    float mass;
    float radius;
    gvec3_t color;
    int num;
    constexpr sphere_proto(float spring, float damping, float shear, float mass, float radius,
                           const gvec3_t& color, int num)
        : spring(spring),
          damping(damping),
          shear(shear),
          mass(mass),
          radius(radius),
          color(color),
          num(num) {}
};

/**
 * @brief 空间中的一个球体, 包含位置/速度/加速度等信息
 *
 */
struct sphere {
    // sphere_proto proto;
    size_t type;
    gvec3_t pos;
    gvec3_t veloc;
    gvec3_t accel;
};

constexpr int sphere_proto_num = 4;
const std::array<sphere_proto, sphere_proto_num> sphere_protos = {
    sphere_proto{.3F, 0.02, 0.1F, 0.02F, 1.F / 16.F, {0.99F, 0.40F, 0.40F}, 10},
    {1.F, 0.02F, 0.1F, 0.015F, 1.F / 24.F, {0.60F, 0.99F, 0.60F}, 500},
    {.8F, 0.02F, 0.1F, 0.01F, 1.F / 32.F, {0.68F, 0.93F, 0.93F}, 100},
    {.8F, 0.02F, 0.1F, 0.01F, 1.F / 48.F, {0.99F, 0.89F, 0.71F}, 3000},
};

#endif  // SPHERE_HPP
