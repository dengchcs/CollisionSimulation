#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "simulator_impl.cuh"

class simulator {
    sphere_proto_arr_t sphere_protos_;
    sphere *spheres_ = nullptr;
    // 一些辅助碰撞检测的数据
    size_t *hashes_ = nullptr;
    size_t *indices_ = nullptr;
    size_t *cell_start_ = nullptr;
    size_t *cell_end_ = nullptr;

    sim_params sim_params_;

    void set_initial_state();
    void init_memory();
    void free_memory();

    void init_sim_params(const char *config_path);

public:
    simulator(const sphere_proto_arr_t &protos, const char *config_path);

    ~simulator();

    /**
     * @brief 获取update()后的球体状态
     */
    [[nodiscard]] auto spheres() const -> const sphere * { return spheres_; }
    [[nodiscard]] auto sphere_num() const -> int { return sim_params_.num_spheres; }

    /**
     * @brief 对所有球体做碰撞检测并更新其状态
     */
    void update(float elapse);
};

#endif  // SIMULATOR_HPP