#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <cstddef>

#include "simulator_impl.cuh"

class simulator {
    sphere *spheres_ = nullptr;
    size_t *hashes_ = nullptr;
    size_t *indices_ = nullptr;
    size_t *cell_start_ = nullptr;
    size_t *cell_end_ = nullptr;

    sim_params sim_params_;

    void set_initial_state();
    void init_memory();
    void free_memory();

    void init_sim_params();

public:
    simulator();

    ~simulator();

    [[nodiscard]] auto spheres() const -> const sphere * { return spheres_; }
    [[nodiscard]] auto sphere_num() const -> int { return sim_params_.num_spheres; }

    void update(float elapse);
};

#endif  // SIMULATOR_HPP