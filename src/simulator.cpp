﻿#include "simulator.hpp"

#include <stddef.h>

#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "GLFW/glfw3.h"
#include "common.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

simulator::simulator() {
    init_window();
    init_shaders();
    init_sphere();
}

void simulator::init_window() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    window_ = glfwCreateWindow(g_win_width, g_win_height, "CollisionSimulation", nullptr, nullptr);
    glfwMakeContextCurrent(window_);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glViewport(0, 0, g_win_width, g_win_height);
    glEnable(GL_DEPTH_TEST);

    std::cerr << "init_window(): " << glGetError() << '\n';
}

void simulator::init_shaders() {
    auto check_shader = [&](GLuint shader, const char* msg) {
        GLint success;
        char log[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, nullptr, log);
            std::cerr << msg << " shader compiling error: " << log << '\n';
        }
    };

    auto check_program = [&](GLuint program, const char* msg) {
        GLint success;
        char log[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, nullptr, log);
            std::cerr << msg << " program link error: " << log << '\n';
        }
    };

    auto load_shader = [&](const char* vert_path, const char* frag_path) -> GLuint {
        std::ifstream vert_file{vert_path}, frag_file{frag_path};
        std::string _vert_code{std::istreambuf_iterator<char>{vert_file}, {}};
        std::string _frag_code{std::istreambuf_iterator<char>{frag_file}, {}};
        auto vert_code = _vert_code.c_str(), frag_code = _frag_code.c_str();

        const auto vert = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 1, &vert_code, nullptr);
        glCompileShader(vert);
        check_shader(vert, vert_path);

        const auto frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &frag_code, nullptr);
        glCompileShader(frag);
        check_shader(frag, frag_path);

        const auto program = glCreateProgram();
        glAttachShader(program, vert);
        glAttachShader(program, frag);
        glLinkProgram(program);
        check_program(program, vert_path);

        glDeleteShader(vert);
        glDeleteShader(frag);

        return program;
    };

    // @todo: 改为可配置模式
    shader_sphere_ = load_shader("./shaders/sphere.vert", "./shaders/phong.frag");

    std::cerr << "init_shaders(): " << glGetError() << '\n';
}

/**
 * @brief 使用三角面片初始化一个标准球面, 将数据写入缓冲中
 */
void simulator::init_sphere() {
    std::vector<gvec3_t> vertices;
    std::vector<gvec3_t> normals;
    std::vector<GLuint> indices;

    constexpr size_t vertical_frag = 64;
    constexpr size_t horizontal_frag = 64;

    for (size_t y = 0; y <= vertical_frag; y++) {
        for (size_t x = 0; x <= horizontal_frag; x++) {
            const float xSeg = (float)x / (float)horizontal_frag;
            const float ySeg = (float)y / (float)vertical_frag;
            const float xPos = std::cos(xSeg * 2.0F * g_pi) * std::sin(ySeg * g_pi);
            const float yPos = std::cos(ySeg * g_pi);
            const float zPos = std::sin(xSeg * 2.0F * g_pi) * std::sin(ySeg * g_pi);
            vertices.emplace_back(xPos, yPos, zPos);
            normals.emplace_back(xPos, yPos, zPos);
        }
    }

    bool odd_row = false;
    for (size_t y = 0; y < vertical_frag; y++) {
        if (!odd_row) {
            for (size_t x = 0; x <= horizontal_frag; x++) {
                indices.push_back(y * (horizontal_frag + 1) + x);
                indices.push_back((y + 1) * (horizontal_frag + 1) + x);
            }
        } else {
            for (int x = horizontal_frag; x >= 0; x--) {
                indices.push_back((y + 1) * (horizontal_frag + 1) + x);
                indices.push_back(y * (horizontal_frag + 1) + x);
            }
        }
        odd_row = !odd_row;
    }

    sphere_indice_cnt_ = indices.size();

    std::vector<float> draw_data;
    draw_data.reserve(3ULL * 2 * vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        draw_data.push_back(vertices[i].x);
        draw_data.push_back(vertices[i].y);
        draw_data.push_back(vertices[i].z);

        draw_data.push_back(normals[i].x);
        draw_data.push_back(normals[i].y);
        draw_data.push_back(normals[i].z);
    }

    glGenVertexArrays(1, &vao_sphere_);
    glGenBuffers(1, &vbo_sphere_);
    glGenBuffers(1, &ebo_sphere_);

    glBindVertexArray(vao_sphere_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_);
    glBufferData(GL_ARRAY_BUFFER, draw_data.size() * sizeof(float), draw_data.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_sphere_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(),
                 GL_STATIC_DRAW);
    const size_t stride = (3 + 3) * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    std::cerr << "init_sphere(): " << glGetError() << '\n';
}

void simulator::draw_spheres() {
    glUseProgram(shader_sphere_);
    glBindVertexArray(vao_sphere_);
    upd_camera(shader_sphere_);

    gmat4_t identity{1.F};    
    glUniformMatrix4fv(glGetUniformLocation(shader_sphere_, "model"), 1, GL_FALSE, &identity[0][0]);
    glUniform1f(glGetUniformLocation(shader_sphere_, "radius"), 1.F);

    glDrawElements(GL_TRIANGLE_STRIP, sphere_indice_cnt_, GL_UNSIGNED_INT, 0);
}

void simulator::loop() {
    while (!glfwWindowShouldClose(window_)) {
        glClearColor(.2F, .3F, .3F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        process_input();

        draw_spheres();

        glfwPollEvents();
        glfwSwapBuffers(window_);

        auto error = glGetError();
        if (error != 0) {
            std::cerr << error << '\n';
        }
    }

    glfwTerminate();
}

void simulator::upd_camera(GLuint shader) {
    glUseProgram(shader);
    const auto view = camera_.view_matrix();
    auto loc = glGetUniformLocation(shader, "view");
    glUniformMatrix4fv(loc, 1, GL_FALSE, &view[0][0]);
    const auto projection =
        glm::perspective(glm::radians(90.F), (float)g_win_width / (float)g_win_height, 0.1F, 10.F);
    loc = glGetUniformLocation(shader, "projection");
}

void simulator::process_input() {
    constexpr float diff = 0.01F;
    if (glfwGetKey(window_, GLFW_KEY_A)) {
        camera_.translate_left(diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_D)) {
        camera_.translate_left(-diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_W)) {
        camera_.translate_up(diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_S)) {
        camera_.translate_up(-diff);
    }
}
