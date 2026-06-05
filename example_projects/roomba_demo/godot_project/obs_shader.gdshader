shader_type spatial;
render_mode unshaded;

uniform float dirt_mask = 0.0;
uniform sampler2D trail_texture : hint_default_black, filter_nearest;

void fragment() {
    // Linearizing depth with 0=near plane, 1=far plane:
	float far = PROJECTION_MATRIX[3][2] / PROJECTION_MATRIX[2][2];
    vec3 ndc = vec3( SCREEN_UV * 2.0 - 1.0, FRAGCOORD.z);
    vec4 view_pos = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
    view_pos /= view_pos.w;
    float linear_depth = -view_pos.z;

    float trail = texture(trail_texture, UV).r;
    ALBEDO = vec3(linear_depth / far, dirt_mask, trail);
}
