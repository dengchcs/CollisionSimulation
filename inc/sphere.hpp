#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <array>

#include "common.hpp"

struct sphere_proto {
    float radius;
    gvec3_t color;
    int num;
    constexpr sphere_proto(float radius, const gvec3_t& color, int num)
        : radius(radius), color(color), num(num) {}
};

constexpr int sphere_proto_num = 4;
const std::array<sphere_proto, sphere_proto_num> sphere_protos = {
    sphere_proto{1.F / 32.F, {0.8F, 0.2F, 0.2F}, 20},
    {1.F / 32.F, {0.2F, 0.8F, 0.2F}, 40},
    {1.F / 48.F, {0.2F, 0.2F, 0.8F}, 60},
    {1.F / 64.F, {0.5F, 0.5F, 0.5F}, 100},
};

#endif  // SPHERE_HPP
