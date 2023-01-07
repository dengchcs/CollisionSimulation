#include "sphere.hpp"

#include <iostream>

#include "toml.hpp"

auto parse_vec(const toml::array &arr, int start, int cnt) -> std::vector<float> {
    std::vector<float> vec(cnt);
    for (int i = start; i < start + cnt; i++) {
        vec[i - start] = arr[i].value<float>().value();
    }
    return vec;
}

auto sphere_proto::parse(const toml::array &proto) -> sphere_proto {
    const float spring = proto[0].value<float>().value();
    const float damping = proto[1].value<float>().value();
    const float shear = proto[2].value<float>().value();
    const float mass = proto[3].value<float>().value();
    const float radius = proto[4].value<float>().value();
    const auto vcolor = parse_vec(*proto[5].as_array(), 0, 3);
    const gvec3_t color = {vcolor[0], vcolor[1], vcolor[2]};
    const int num = proto[6].value<int>().value();
    return {spring, damping, shear, mass, radius, color, num};
}

auto parse_protos(const char *name) -> sphere_proto_arr_t {
    const auto config = *toml::parse_file(name).get_as<toml::array>("sphere-protos");
    if (config.size() != sphere_proto_num) {
        fprintf(stderr, "should have exactly %d protos\n", sphere_proto_num);
        exit(1);
    }
    std::array<sphere_proto, sphere_proto_num> protos;
    for (int i = 0; i < sphere_proto_num; i++) {
        protos[i] = sphere_proto::parse(*config[i].as_array());
    }
    return protos;
}
