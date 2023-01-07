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
    camera() : eye_(2, 1, 2), up_(glm::normalize(gvec3_t{-1, 3, -1})), center_(-1, -1, -1){};

    camera(const gvec3_t& eye, const gvec3_t& up, const gvec3_t& center)
        : eye_(eye),
          up_(glm::normalize(up)),
          center_(center){

          };

    [[nodiscard]] auto eye_position() const { return eye_; }

    [[nodiscard]] auto view_matrix() const -> gmat4_t { return glm::lookAt(eye_, center_, up_); }

    /**
     * @brief 相机向上下方向移动
     *
     * @param dist 为正值时向上移动, 为负值时向下移动
     */
    void translate_up(float dist) {
        eye_ += dist * up_;
        center_ += dist * up_;
    }

    /**
     * @brief 相机向左右移动
     *
     * @param dist 为正值时向左移动, 为负值时向右移动
     */
    void translate_left(float dist) {
        auto z_axis = glm::normalize(center_ - eye_);
        auto x_axis = glm::normalize(glm::cross(up_, z_axis));
        eye_ += dist * x_axis;
        center_ += dist * x_axis;
    }

    /**
     * @brief 相机向前后移动
     *
     * @param dist 为正值时向前移动, 为负值时向后移动
     */
    void translate_forward(float dist) {
        auto z_axis = glm::normalize(center_ - eye_);
        eye_ += dist * z_axis;
        center_ += dist * z_axis;
    }

    /**
     * @brief 相机绕视点向上下旋转
     *
     * @param degree 角度制. 为正值时向上旋转, 为负值时向下旋转
     */
    void rotate_up(float degree) {
        auto tmp_eye = eye_ - center_;
        auto x = glm::normalize(glm::cross(up_, tmp_eye));
        tmp_eye = glm::rotate(gmat4_t{1.F}, glm::radians(degree), x) * gvec4_t{tmp_eye, 1.F};
        up_ = glm::normalize(glm::cross(tmp_eye, x));
        eye_ = tmp_eye + center_;
    }

    /**
     * @brief 相机绕视点水平旋转
     *
     * @param degree 角度制. 为正值时向左旋转, 为负值时向右旋转
     */
    void rotate_left(float degree) {
        auto tmp_eye = eye_ - center_;
        tmp_eye = glm::rotate(gmat4_t{1.F}, glm::radians(degree), up_) * gvec4_t{tmp_eye, 1.F};
        tmp_eye = tmp_eye + center_;
        eye_ = tmp_eye;
    }
};

#endif