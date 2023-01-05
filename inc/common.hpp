#ifndef COMMON_HPP
#define COMMON_HPP

#include <glad/glad.h>

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>


using gvec3_t = glm::vec3;
using gmat4_t = glm::mat4;
using gvec4_t = glm::vec4;
using gvec3i_t = glm::vec<3, int>;

constexpr GLuint g_win_width = 800;
constexpr GLuint g_win_height = 600;

constexpr float g_pi = 3.14159265F;

#endif