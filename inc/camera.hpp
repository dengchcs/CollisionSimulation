#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/gtc/matrix_transform.hpp>

#include "common.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/geometric.hpp"

class camera {
    gvec3_t eye_;
    gvec3_t up_;
    gvec3_t center_;

public:
    camera() : eye_(0, 0, 1), up_(0, 1, 0), center_(0, 0, 0){};

    camera(const gvec3_t& eye, const gvec3_t& up, const gvec3_t& center)
        : eye_(eye),
          up_(glm::normalize(up)),
          center_(center){

          };

    [[nodiscard]] auto eye_position() const { return eye_; }

    [[nodiscard]] auto view_matrix() const -> gmat4_t { return glm::lookAt(eye_, center_, up_); }

    void translate_up(float dist) {
        eye_ += dist * up_;
        center_ += dist * up_;
    }

    void translate_left(float dist) {
        auto z_axis = glm::normalize(center_ - eye_);
        auto x_axis = glm::normalize(glm::cross(up_, z_axis));
        eye_ += dist * x_axis;
        center_ += dist * x_axis;
    }

    void translate_forward(float dist) {
        auto z_axis = glm::normalize(center_ - eye_);

        eye_ += dist * z_axis;
        center_ += dist * z_axis;
    }
};

#endif