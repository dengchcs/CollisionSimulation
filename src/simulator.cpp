#include "simulator.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

#include "GLFW/glfw3.h"
#include "common.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "shader.hpp"
#include "sphere.hpp"

simulator::simulator() {
    init_window();
    init_shader();
    init_sphere();
    init_walls();
    engine_ = new engine{};
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

void simulator::init_shader() {
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

    auto load_shader = [&](const char* vert_code, const char* frag_code) -> GLuint {
        const auto vert = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 1, &vert_code, nullptr);
        glCompileShader(vert);
        check_shader(vert, "vert shader");

        const auto frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &frag_code, nullptr);
        glCompileShader(frag);
        check_shader(frag, "frag shader");

        const auto program = glCreateProgram();
        glAttachShader(program, vert);
        glAttachShader(program, frag);
        glLinkProgram(program);
        check_program(program, "program");

        glDeleteShader(vert);
        glDeleteShader(frag);

        return program;
    };

    shader_ = load_shader(g_phong_vert, g_phong_frag);

    std::cerr << "init_shaders(): " << glGetError() << '\n';
}

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

void simulator::init_walls() {
    // clang-format off
    float walls[] = {
        // 左侧面
        -1, -1, -1, 1, 0, 0,
        -1, -1, +1, 1, 0, 0,
        -1, +1, -1, 1, 0, 0,
        -1, +1, +1, 1, 0, 0,
        -1, -1, +1, 1, 0, 0,
        -1, +1, -1, 1, 0, 0,
        // 后面
        -1, -1, -1, 0, 0, 1,
        -1, +1, -1, 0, 0, 1,
        +1, -1, -1, 0, 0, 1,
        +1, +1, -1, 0, 0, 1,
        -1, +1, -1, 0, 0, 1,
        +1, -1, -1, 0, 0, 1,
        // 底面
        -1, -1, -1, 0, 1, 0,
        -1, -1, +1, 0, 1, 0,
        +1, -1, -1, 0, 1, 0,
        +1, -1, +1, 0, 1, 0,
        -1, -1, +1, 0, 1, 0,
        +1, -1, -1, 0, 1, 0,
    };
    // clang-format on
    glGenVertexArrays(1, &vao_walls_);
    glGenBuffers(1, &vbo_walls_);

    glBindVertexArray(vao_walls_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_walls_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(walls), walls, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void simulator::draw_spheres() {
    glUseProgram(shader_);
    glBindVertexArray(vao_sphere_);
    upd_scene();

    const float* pos = engine_->sphere_pos();
    const int* type = engine_->sphere_type();
    const int num = engine_->sphere_num();
    for (int i = 0; i < num; i++) {
        const auto proto = sphere_protos[type[i]];
        glUniform1f(glGetUniformLocation(shader_, "scale"), proto.radius);
        glUniform3fv(glGetUniformLocation(shader_, "objectColor"), 1, &(proto.color[0]));
        const gvec3_t sphere_pos{pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
        const auto model = glm::translate(gmat4_t{1.F}, sphere_pos);
        glUniformMatrix4fv(glGetUniformLocation(shader_, "model"), 1, GL_FALSE, &model[0][0]);
        glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)sphere_indice_cnt_, GL_UNSIGNED_INT, 0);
    }
}

void simulator::draw_walls() {
    glUseProgram(shader_);
    glBindVertexArray(vao_walls_);
    upd_scene();
    gmat4_t identity{1.F};
    glUniformMatrix4fv(glGetUniformLocation(shader_, "model"), 1, GL_FALSE, &identity[0][0]);
    glUniform1f(glGetUniformLocation(shader_, "scale"), 1.0F);
    glUniform3f(glGetUniformLocation(shader_, "objectColor"), 0.1F, 0.1F, 0.6F);

    glDrawArrays(GL_TRIANGLES, 0, 18);
}

void simulator::loop() {
    constexpr double fps = 1.0 / 30.0;
    double last_update_time = 0;
    double last_frame_time = 0;

    while (!glfwWindowShouldClose(window_)) {
        const double now = glfwGetTime();

        glfwPollEvents();
        glClearColor(.1F, .1F, .1F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        process_input();

        draw_spheres();
        draw_walls();

        auto error = glGetError();
        if (error != 0) {
            std::cerr << error << '\n';
        }

        if (now - last_frame_time >= fps) {
            glfwSwapBuffers(window_);
            last_frame_time = now;
        }
        last_update_time = now;
    }

    glfwTerminate();
}

void simulator::upd_scene() {
    glUseProgram(shader_);

    const auto view = camera_.view_matrix();
    auto loc = glGetUniformLocation(shader_, "view");
    glUniformMatrix4fv(loc, 1, GL_FALSE, &view[0][0]);

    const auto projection =
        glm::perspective(glm::radians(90.F), (float)g_win_width / (float)g_win_height, 0.1F, 10.F);
    loc = glGetUniformLocation(shader_, "projection");
    glUniformMatrix4fv(loc, 1, GL_FALSE, &projection[0][0]);

    loc = glGetUniformLocation(shader_, "lightPos");
    glUniform3f(loc, 0.F, 1.F, 0.F);

    loc = glGetUniformLocation(shader_, "viewPos");
    const auto cam_pos = camera_.eye_position();
    glUniform3f(loc, cam_pos.x, cam_pos.y, cam_pos.z);
}

void simulator::process_input() {
    constexpr float move_diff = 0.001F;
    if (glfwGetKey(window_, GLFW_KEY_A)) {
        camera_.translate_left(move_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_D)) {
        camera_.translate_left(-move_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_W)) {
        camera_.translate_up(move_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_S)) {
        camera_.translate_up(-move_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_Z)) {
        camera_.translate_forward(move_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_X)) {
        camera_.translate_forward(-move_diff);
    }

    constexpr float rot_diff = 5.F;
    if (glfwGetKey(window_, GLFW_KEY_UP)) {
        camera_.rotate_up(rot_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_DOWN)) {
        camera_.rotate_up(-rot_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_LEFT)) {
        camera_.rotate_left(rot_diff);
    }
    if (glfwGetKey(window_, GLFW_KEY_RIGHT)) {
        camera_.rotate_left(-rot_diff);
    }
}
