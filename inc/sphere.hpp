#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <array>
#include "common.hpp"

class sphere {
    gvec3_t center_;
    float radius_;
    gvec3_t color_;

public:
    constexpr sphere(const gvec3_t& center, float radius, const gvec3_t& color)
        : center_(center), radius_(radius), color_(color) {}
};

constexpr int proto_num = 4;
const std::array<sphere, proto_num> protos = {
    sphere{{0, 0, 0}, 1.F / 32.F, {0.8F, 0.2F, 0.2F}},
    {{0, 0, 0}, 1.F / 32.F, {0.2F, 0.8F, 0.2F}},
    {{0, 0, 0}, 1.F / 48.F, {0.2F, 0.2F, 0.8F}},
    {{0, 0, 0}, 1.F / 64.F, {0.5F, 0.5F, 0.5F}},
};

#endif  // SPHERE_HPP
