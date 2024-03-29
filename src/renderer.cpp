﻿#include "renderer.hpp"

#include <iostream>
#include <vector>

#include "GLFW/glfw3.h"
#include "common.hpp"
#include "glm/ext/matrix_clip_space.hpp"  // glm::perspective()
#include "glm/ext/matrix_transform.hpp"   // glm::translate()
#include "shader.hpp"
#include "sphere.hpp"
#include "toml.hpp"

auto renderer::config::parse(const char* file) -> config {
    const auto data = toml::parse_file(file)["renderer"];
    config parsed;
    parsed.width = data["width"].value_or(800);
    parsed.height = data["height"].value_or(800);
    parsed.fps = data["fps"].value_or(30);
    parsed.sphere_frag = data["sphere_frag"].value_or(64);
    parsed.move_diff = data["move_diff"].value_or(0.005F);
    parsed.rot_diff = data["rot_diff"].value_or(1.0F);
    return parsed;
}

renderer::renderer(const char* config_path) {
    sphere_protos_ = sphere_proto::parse(config_path);
    config_ = config::parse(config_path);
    simulator_ = new simulator{sphere_protos_, config_path};
    // 先初始化配置项再初始化gl
    init_window();
    init_shader();
    init_sphere();
    init_walls();
}

renderer::~renderer() {
    delete simulator_;
    glfwTerminate();
}

void renderer::init_window() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    window_ =
        glfwCreateWindow(config_.width, config_.height, "CollisionSimulation", nullptr, nullptr);
    glfwMakeContextCurrent(window_);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glViewport(0, 0, config_.width, config_.height);
    glEnable(GL_DEPTH_TEST);

    // std::cerr << "init_window(): " << glGetError() << '\n';
}

void renderer::init_shader() {
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

    // 程序只使用了一个shader程序, 但抽象成lambda便于后续扩展
    shader_ = load_shader(g_phong_vert, g_phong_frag);

    // std::cerr << "init_shaders(): " << glGetError() << '\n';
}

void renderer::init_sphere() {
    // 球面由多个三角面片近似而成

    std::vector<gvec3_t> vertices;
    std::vector<gvec3_t> normals;
    std::vector<GLuint> indices;

    for (size_t y = 0; y <= config_.sphere_frag; y++) {
        for (size_t x = 0; x <= config_.sphere_frag; x++) {
            const float xSeg = (float)x / (float)config_.sphere_frag;
            const float ySeg = (float)y / (float)config_.sphere_frag;
            const float xPos = std::cos(xSeg * 2.0F * g_pi) * std::sin(ySeg * g_pi);
            const float yPos = std::cos(ySeg * g_pi);
            const float zPos = std::sin(xSeg * 2.0F * g_pi) * std::sin(ySeg * g_pi);
            vertices.emplace_back(xPos, yPos, zPos);
            normals.emplace_back(xPos, yPos, zPos);
        }
    }

    bool odd_row = false;
    for (size_t y = 0; y < config_.sphere_frag; y++) {
        if (!odd_row) {
            for (size_t x = 0; x <= config_.sphere_frag; x++) {
                indices.push_back(y * (config_.sphere_frag + 1) + x);
                indices.push_back((y + 1) * (config_.sphere_frag + 1) + x);
            }
        } else {
            for (int x = config_.sphere_frag; x >= 0; x--) {
                indices.push_back((y + 1) * (config_.sphere_frag + 1) + x);
                indices.push_back(y * (config_.sphere_frag + 1) + x);
            }
        }
        odd_row = !odd_row;
    }

    sphere_indice_cnt_ = indices.size();

    // 和 vert shader 保持一致: 输入是两个 vec3: pos & normal
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

    // std::cerr << "init_sphere(): " << glGetError() << '\n';
}

void renderer::init_walls() {
    // clang-format off
    // 墙面由若干三角形组成, 每行数据: 顶点 + 法向
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

void renderer::draw_spheres() {
    glUseProgram(shader_);
    glBindVertexArray(vao_sphere_);
    upd_scene();

    const auto* spheres = simulator_->spheres();
    const int num = simulator_->sphere_num();
    for (int i = 0; i < num; i++) {
        const auto proto = sphere_protos_[spheres[i].type];
        glUniform1f(glGetUniformLocation(shader_, "scale"), proto.radius);
        glUniform3fv(glGetUniformLocation(shader_, "objectColor"), 1, &(proto.color[0]));
        // "标准"球体是在原点, 所以绘制时的平移量就是球心位置
        const auto model = glm::translate(gmat4_t{1.F}, spheres[i].pos);
        glUniformMatrix4fv(glGetUniformLocation(shader_, "model"), 1, GL_FALSE, &model[0][0]);
        glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)sphere_indice_cnt_, GL_UNSIGNED_INT, 0);
    }
}

void renderer::draw_walls() {
    glUseProgram(shader_);
    glBindVertexArray(vao_walls_);
    upd_scene();
    gmat4_t identity{1.F};
    glUniformMatrix4fv(glGetUniformLocation(shader_, "model"), 1, GL_FALSE, &identity[0][0]);
    glUniform1f(glGetUniformLocation(shader_, "scale"), 1.0F);
    glUniform3f(glGetUniformLocation(shader_, "objectColor"), 0.79F, 0.79F, 0.99F);

    glDrawArrays(GL_TRIANGLES, 0, 18);
}

void renderer::loop() {
    double last_update_time = glfwGetTime();
    double last_frame_time = last_update_time;

    while (!glfwWindowShouldClose(window_)) {
        const double now = glfwGetTime();
        const auto elapse = (float)(now - last_update_time);

        glfwPollEvents();
        glClearColor(.75F, .75F, .75F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        process_input();

        simulator_->update(elapse);
        draw_spheres();
        draw_walls();

        auto error = glGetError();
        if (error != 0) {
            std::cerr << error << '\n';
        }

        // 控制帧率
        if (now - last_frame_time >= 1.0F / (float)config_.fps) {
            glfwSwapBuffers(window_);
            last_frame_time = now;
        }
        last_update_time = now;
    }

    // glfwTerminate();
}

void renderer::upd_scene() {
    glUseProgram(shader_);

    const auto view = camera_.view_matrix();
    auto loc = glGetUniformLocation(shader_, "view");
    glUniformMatrix4fv(loc, 1, GL_FALSE, &view[0][0]);

    const auto projection = glm::perspective(
        glm::radians(60.F), (float)config_.width / (float)config_.height, 0.1F, 100.F);
    loc = glGetUniformLocation(shader_, "projection");
    glUniformMatrix4fv(loc, 1, GL_FALSE, &projection[0][0]);

    loc = glGetUniformLocation(shader_, "lightPos");
    glUniform3f(loc, 1.F, 1.F, 1.F);  // 光照设置在距三面墙的最远处

    loc = glGetUniformLocation(shader_, "viewPos");
    const auto cam_pos = camera_.eye_position();
    glUniform3f(loc, cam_pos.x, cam_pos.y, cam_pos.z);
}

void renderer::process_input() {
    const float move_diff = config_.move_diff;
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

    const float rot_diff = config_.rot_diff;
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
