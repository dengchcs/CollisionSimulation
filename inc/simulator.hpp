#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

class simulator {
    float *sphere_pos_ = nullptr;    // 每个球的XYZ坐标
    float *sphere_velo_ = nullptr;   // 每个球XYZ方向的速度
    float *sphere_accel_ = nullptr;  // 每个球XYZ方向的加速度
    int *sphere_type_ = nullptr;     // 每个球的类型(在protos中的下标)
    float max_radius_ = 0;           // 几种球中最大的半径
    int sphere_num_ = 0;             // 要模拟的球的个数

    void set_initial_state();
    void init_memory();
    void free_memory();

public:
    simulator();

    ~simulator();

    [[nodiscard]] auto sphere_pos() const -> const float * { return sphere_pos_; }
    [[nodiscard]] auto sphere_type() const -> const int * { return sphere_type_; }
    [[nodiscard]] auto sphere_num() const -> int { return sphere_num_; }

    void update(float elapse);
};

#endif  // SIMULATOR_HPP