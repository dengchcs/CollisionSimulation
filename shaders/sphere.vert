﻿#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 fragPos;
out vec3 normal;

uniform float radius;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    float nouse = radius;
    fragPos = vec3(model * vec4(aPos * radius, 1.0f));
    normal = mat3(transpose(inverse(model))) * aNormal;
    vec4 pos = projection * view * vec4(fragPos, 1.0F);
    gl_Position = projection * view * vec4(fragPos, 1.0F);
}