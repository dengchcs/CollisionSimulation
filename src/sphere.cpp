#include "sphere.hpp"

#include <iostream>

#include "toml.hpp"

auto parse_vec(const toml::array &arr, int start, int cnt, float defaul) -> std::vector<float> {
    std::vector<float> vec(cnt);
    for (int i = start; i < start + cnt; i++) {
        vec[i - start] = arr[i].value_or(defaul);
    }
    return vec;
}

auto parse_single(const toml::array &proto) -> sphere_proto {
    const float spring = proto[0].value_or(0.8F);
    const float damping = proto[1].value_or(0.02F);
    const float shear = proto[2].value_or(0.1F);
    const float mass = proto[3].value_or(0.04F);
    const float radius = proto[4].value_or(0.05F);
    const auto vcolor = parse_vec(*proto[5].as_array(), 0, 3, 1.0F);
    const gvec3_t color = {vcolor[0], vcolor[1], vcolor[2]};
    const int num = proto[6].value_or(1);
    return {spring, damping, shear, mass, radius, color, num};
}

auto sphere_proto::parse(const char *file) -> sphere_proto_arr_t {
    const auto config = *toml::parse_file(file).get_as<toml::array>("sphere-protos");
    if (config.size() != sphere_proto_num) {
        fprintf(stderr, "should have exactly %d protos\n", sphere_proto_num);
        exit(1);
    }
    std::array<sphere_proto, sphere_proto_num> protos;
    for (int i = 0; i < sphere_proto_num; i++) {
        protos[i] = parse_single(*config[i].as_array());
    }
    return protos;
}
