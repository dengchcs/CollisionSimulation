#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <array>
#include <cstddef>

#include "common.hpp"

/**
 * @brief 球体的"原型", 不包含运动信息
 */
struct sphere_proto {
    float mass;
    float radius;
    gvec3_t color;
    int num;
    constexpr sphere_proto(float mass, float radius, const gvec3_t& color, int num)
        : mass(mass), radius(radius), color(color), num(num) {}
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
    sphere_proto{2.0F, 1.F / 16.F, {0.99F, 0.40F, 0.40F}, 10},
    {1.5F, 1.F / 16.F, {0.60F, 0.99F, 0.60F}, 10},
    {1.F, 1.F / 24.F, {0.68F, 0.93F, 0.93F}, 10},
    {1.F, 1.F / 32.F, {0.99F, 0.89F, 0.71F}, 10},
};

#endif  // SPHERE_HPP
