// GLSL source for the copy_viewports compute shader, embedded as a raw string
// literal so the extension has no runtime file dependency.
//
// The shader copies one SubViewport's RGBA8 color attachment into the flat
// staging buffer. The caller dispatches once per env, updating the uniform set
// (and therefore the source texture binding) between dispatches.
//
// When editing this shader, this file is the canonical source — there is no
// separate .glsl file.

const char *k_copy_viewports_glsl = R"glsl(
#version 450

// Requires VK_KHR_shader_float16_int8 (universally available on Vulkan 1.1+).
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// One dispatch covers one environment. env_index selects the output slot in
// the flat staging buffer; the caller steps through envs in a loop.
layout(push_constant, std430) uniform PushConstants {
    uint env_index;
    uint width;
    uint height;
    uint out_channels; // 1-4; selects how many RGBA channels to write
} pc;

// Source: one SubViewport color attachment, bound per-env by the caller.
layout(set = 0, binding = 0) uniform sampler2D src_tex;

// Destination: shared staging buffer covering all envs.
// Layout: [env0 pixels][env1 pixels]..., each pixel is out_channels bytes.
layout(set = 0, binding = 1, std430) writeonly buffer OutputBuffer {
    uint8_t data[];
} out_buf;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    // Discard threads that fall outside the viewport (when dims aren't multiples of 8).
    if (x >= pc.width || y >= pc.height) return;

    // texelFetch skips filtering - we want raw pixel values.
    vec4 color = texelFetch(src_tex, ivec2(x, y), 0);

    // Convert [0, 1] floats to [0, 255] bytes. clamp guards against out-of-range
    // values that some renderers can produce (e.g. HDR overflow or precision drift).
    u8vec4 rgba = u8vec4(
        uint8_t(clamp(uint(color.r * 255.0 + 0.5), 0u, 255u)),
        uint8_t(clamp(uint(color.g * 255.0 + 0.5), 0u, 255u)),
        uint8_t(clamp(uint(color.b * 255.0 + 0.5), 0u, 255u)),
        uint8_t(clamp(uint(color.a * 255.0 + 0.5), 0u, 255u))
    );

    uint pixel_base = (pc.env_index * pc.width * pc.height + y * pc.width + x) * pc.out_channels;

    // Write only the requested channels. The branches are uniform across the
    // workgroup (push constant), so the GPU folds them at wave level.
    if (pc.out_channels > 0u) out_buf.data[pixel_base + 0u] = rgba.r;
    if (pc.out_channels > 1u) out_buf.data[pixel_base + 1u] = rgba.g;
    if (pc.out_channels > 2u) out_buf.data[pixel_base + 2u] = rgba.b;
    if (pc.out_channels > 3u) out_buf.data[pixel_base + 3u] = rgba.a;
}
)glsl";
