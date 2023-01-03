#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <stddef.h>

#include "GLFW/glfw3.h"
#include "camera.hpp"
#include "glad/glad.h"


class simulator {
    camera camera_{};

    GLFWwindow *window_ = nullptr;

    GLuint shader_sphere_ = 0;
    GLuint vao_sphere_ = 0;
    GLuint vbo_sphere_ = 0;
    GLuint ebo_sphere_ = 0;

    size_t sphere_indice_cnt_ = 0;

    void init_window();
    void init_shaders();

    void init_sphere();
    void draw_spheres();
    void draw_triangle();

    void upd_camera(GLuint shader);

    void process_input();

public:
    simulator();
    ~simulator() = default;
    void loop();
};

#endif