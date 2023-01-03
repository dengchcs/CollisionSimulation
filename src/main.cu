#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <iostream>

#include "simulator.hpp"

auto main(int argc, char** argv) -> int {
    simulator demo;
    demo.loop();
    return 0;
}
