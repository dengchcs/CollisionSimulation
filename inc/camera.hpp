#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "common.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/geometric.hpp"
#include "glm/trigonometric.hpp"

class camera {
    gvec3_t eye_;
    gvec3_t up_;
    gvec3_t center_;

public:
    camera() : eye_(1, 1, 1), up_(glm::normalize(gvec3_t{-1, 2, -1})), center_(0, 0, 0){};

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

    void rotate_up(float degree) {
        auto tmp_eye = eye_ - center_;
        auto x = glm::normalize(glm::cross(up_, tmp_eye));
        tmp_eye = glm::rotate(gmat4_t{1.F}, glm::radians(degree), x) * gvec4_t{tmp_eye, 1.F};
        up_ = glm::normalize(glm::cross(tmp_eye, x));
        eye_ = tmp_eye + center_;
    }

    void rotate_left(float degree) {
        auto tmp_eye = eye_ - center_;
        tmp_eye = glm::rotate(gmat4_t{1.F}, glm::radians(degree), up_) * gvec4_t{tmp_eye, 1.F};
        tmp_eye = tmp_eye + center_;
        eye_ = tmp_eye;
    }
};

#endif