#include <chrono>

#include "simulator.hpp"
#include "toml.hpp"

auto main(int argc, char **argv) -> int {
    if (argc != 3) {
        fprintf(stderr, "usage: test <config-path> <update-cnt>");
    }
    const auto protos = sphere_proto::parse(argv[1]);
    const int fps = toml::parse_file(argv[1])["renderer"]["fps"].value_or(30);
    const float elapse = 1.0F / (float)fps;
    simulator simu{protos, argv[1]};
    const int sphere_num = simu.sphere_num();
    const int upd_cnt = std::stoi(argv[2]);
    printf("test started: update %d spheres %d times\n", sphere_num, upd_cnt);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < upd_cnt; i++) {
        simu.update(elapse);
    }
    auto end = std::chrono::system_clock::now();
    const auto time = std::chrono::duration<double>(end - start).count();
    printf("test finished: time usage = %fs\n", time);
    return 0;
}
