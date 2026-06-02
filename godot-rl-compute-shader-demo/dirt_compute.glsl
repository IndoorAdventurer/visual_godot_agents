#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    vec4 particles[];
};
layout(set = 0, binding = 1, std430) buffer CounterBuffer {
    uint counter;
};
layout(set = 0, binding = 2, std430) buffer MultiMeshBuffer {
    float transforms[];
};

layout(push_constant, std430) uniform PushConstants {
    vec2 robot_pos;
    float suction_radius;
    float collection_radius;
    float delta;
    uint n_particles;
} pc;

void write_transform(uint i, vec4 p) {
    uint base = i * 12;
    transforms[base + 0]  = p.w; transforms[base + 1]  = 0.0; transforms[base + 2]  = 0.0; transforms[base + 3]  = p.x;
    transforms[base + 4]  = 0.0; transforms[base + 5]  = p.w; transforms[base + 6]  = 0.0; transforms[base + 7]  = p.y + 0.0001;
    transforms[base + 8]  = 0.0; transforms[base + 9]  = 0.0; transforms[base + 10] = p.w; transforms[base + 11] = p.z;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    // Check if we are inside the range particles[]
    if (i >= pc.n_particles) return;
    vec4 p = particles[i];

    // Ignore dead particles
    if (p.w < 0.5) {
        write_transform(i, p);
        return;
    }

    float to_robot = distance(pc.robot_pos, p.xz);
    
    // Kill particles that are close to the vacuum
    if (to_robot < pc.collection_radius) {
        p.w = 0.0;
        atomicAdd(counter, 1u);
    }
    else if (to_robot < pc.suction_radius) {
        float dist2 = to_robot * to_robot;
        float rad2 = pc.suction_radius;
        float pull = (rad2 - dist2) / ( (dist2 + 0.5) * (rad2 + 0.5) );
        p.xz += normalize(pc.robot_pos - p.xz) * pull * pc.delta;
    }

    particles[i] = p;
    write_transform(i, p);
}