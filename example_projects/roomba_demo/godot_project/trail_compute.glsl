#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, r8) uniform image2D trail_tex;

layout(push_constant, std430) uniform PushConstants {
    vec2 robot_uv;      // robot position in UV space [0, 1]
    float brush_radius; // brush radius in UV space
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(trail_tex);
    if (any(greaterThanEqual(coord, size))) return;

    vec2 uv = (vec2(coord) + 0.5) / vec2(size);
    if (distance(uv, pc.robot_uv) < pc.brush_radius)
        imageStore(trail_tex, coord, vec4(1.0));
}
