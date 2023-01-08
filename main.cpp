#include "renderer.hpp"

auto main(int argc, char** argv) -> int {
    if (argc != 2) {
        fprintf(stderr, "usage: collision <config-path>\n");
    }
    renderer demo{argv[1]};
    demo.loop();
    return 0;
}
