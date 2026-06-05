class_name TrailSystem
extends MeshInstance3D

# Set from env.gd
var robot: Node3D

@export var trail_size: int = 128
# Brush radius in UV space — tune to match robot footprint relative to 5m floor
@export var brush_radius: float = 0.03

var _rd: RenderingDevice
var _texture_rid: RID
var _cp_shader: RID
var _pipeline: RID
var _uniform_set: RID

func _ready() -> void:
	_rd = RenderingServer.get_rendering_device()
	_setup_texture()
	_setup_compute()
	reset()

func _physics_process(_delta: float) -> void:
	if robot == null:
		return
	var pos := robot.global_position
	# Convert world XZ [-2.5, 2.5] to UV [0, 1]
	var uv := Vector2((pos.x + 2.5) / 5.0, (pos.z + 2.5) / 5.0)

	#layout(push_constant, std430) uniform PushConstants {
	#    vec2 robot_uv;      8 bytes offset 0
	#    float brush_radius; 4 bytes offset 8
	#    float _pad;         4 bytes offset 12
	#} pc;                   16 bytes total
	var push_consts := PackedByteArray()
	push_consts.resize(16)
	push_consts.encode_float(0, uv.x)
	push_consts.encode_float(4, uv.y)
	push_consts.encode_float(8, brush_radius)
	# _pad left as zero

	var cl := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(cl, _pipeline)
	_rd.compute_list_bind_uniform_set(cl, _uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, push_consts, push_consts.size())
	var groups := ceili(trail_size / 8.0)
	_rd.compute_list_dispatch(cl, groups, groups, 1)
	_rd.compute_list_end()

func reset() -> void:
	_rd.texture_clear(_texture_rid, Color(0, 0, 0, 0), 0, 1, 0, 1)

func _setup_texture() -> void:
	var tf := RDTextureFormat.new()
	tf.format = RenderingDevice.DATA_FORMAT_R8_UNORM
	tf.width = trail_size
	tf.height = trail_size
	tf.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)
	_texture_rid = _rd.texture_create(tf, RDTextureView.new())

	# Create a fresh material per instance to avoid sharing the mesh's material
	# across multiple scene instances.
	var mat := ShaderMaterial.new()
	mat.shader = load("res://obs_shader.gdshader")
	mat.set_shader_parameter("dirt_mask", 0.0)
	var trail_texture := Texture2DRD.new()
	trail_texture.texture_rd_rid = _texture_rid
	mat.set_shader_parameter("trail_texture", trail_texture)
	set_surface_override_material(0, mat)

func _setup_compute() -> void:
	var spirv: RDShaderSPIRV = load("res://trail_compute.glsl").get_spirv()
	_cp_shader = _rd.shader_create_from_spirv(spirv)
	_pipeline = _rd.compute_pipeline_create(_cp_shader)

	var img_uniform := RDUniform.new()
	img_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	img_uniform.binding = 0
	img_uniform.add_id(_texture_rid)

	_uniform_set = _rd.uniform_set_create([img_uniform], _cp_shader, 0)
