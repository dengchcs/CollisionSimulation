#ifndef SYSTEM_HPP
#define SYSTEM_HPP

#include <cstddef>

#include "GLFW/glfw3.h"
#include "camera.hpp"
#include "engine.hpp"

class simulator {
    camera camera_{};

    GLFWwindow *window_ = nullptr;

    GLuint shader_ = 0;

    GLuint vao_sphere_ = 0;
    GLuint vbo_sphere_ = 0;
    GLuint ebo_sphere_ = 0;

    GLuint vao_walls_ = 0;
    GLuint vbo_walls_ = 0;

    size_t sphere_indice_cnt_ = 0;

    engine *engine_ = nullptr;

    void init_window();
    void init_shader();

    /**
     * @brief 使用三角面片初始化一个标准球面, 将数据写入缓冲中
     */
    void init_sphere();
    void draw_spheres();

    /**
     * @brief 初始化左/后/底三面"墙"的数据, 写入缓冲
     */
    void init_walls();
    void draw_walls();

    /**
     * @brief 根据相机状态设置shader中的相关变量
     */
    void upd_scene();

    /**
     * @brief 处理键盘输入事件
     */
    void process_input();

public:
    simulator();
    ~simulator() = default;
    void loop();
};

#endif