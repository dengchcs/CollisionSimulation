#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <array>

#include "common.hpp"

struct sphere_proto;
constexpr int sphere_proto_num = 4;
using sphere_proto_arr_t = std::array<sphere_proto, sphere_proto_num>;

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
    sphere_proto() = default;
    constexpr sphere_proto(float spring, float damping, float shear, float mass, float radius,
                           const gvec3_t& color, int num)
        : spring(spring),
          damping(damping),
          shear(shear),
          mass(mass),
          radius(radius),
          color(color),
          num(num) {}
    static auto parse(const char* file) -> sphere_proto_arr_t;
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

#endif  // SPHERE_HPP
