#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>

#include "renderer.hpp"

auto main(int argc, char** argv) -> int {
    renderer demo;
    demo.loop();
    return 0;
}
