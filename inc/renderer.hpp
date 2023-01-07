#ifndef RENDERER_HPP
#define RENDERER_HPP

#include "GLFW/glfw3.h"
#include "camera.hpp"
#include "simulator.hpp"
#include "sphere.hpp"

class renderer {
    camera camera_{};

    GLFWwindow *window_ = nullptr;

    GLuint shader_ = 0;

    GLuint vao_sphere_ = 0;
    GLuint vbo_sphere_ = 0;
    GLuint ebo_sphere_ = 0;

    GLuint vao_walls_ = 0;
    GLuint vbo_walls_ = 0;

    size_t sphere_indice_cnt_ = 0;  // 记录单个球渲染的面片数量

    simulator *simulator_ = nullptr;
    sphere_proto_arr_t sphere_protos_;

    /**
     * @brief 初始化glfw系统
     */
    void init_window();
    /**
     * @brief 读取shader代码, 创建shader程序
     */
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
    renderer(const char *config_path);
    ~renderer();

    /**
     * @brief 主循环: 计算球体状态并渲染
     */
    void loop();
};

#endif  // RENDERER_HPP