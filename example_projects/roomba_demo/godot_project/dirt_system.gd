class_name DirtSystem
extends Node3D

# Set one level up
var robot: Node3D
var agent: VGAAgentNode

# The number of dirt particles to clean up
@export var n_particles : int = 700

# Standard deviation for particle spread
@export var spread_std : float = 0.4

@export var suction_radius : float = 0.75
@export var collection_radius : float = 0.1

var _rd: RenderingDevice
var _buffer: RID
var _byte_count : int

var _cp_shader : RID
var _pipeline : RID
var _uniform_set : RID

var _mm: MultiMesh
var _mmi3D: MultiMeshInstance3D

func _ready() -> void:
	_rd = RenderingServer.get_rendering_device()
	
	# 4 floats per particle (x, y, z, alive), 4 bytes each
	_byte_count = n_particles * 4 * 4
	_buffer = _rd.storage_buffer_create(_byte_count)
	
	_multimesh_stuff()
	_setup_compute()
	reset()

func _physics_process(delta: float) -> void:
	if robot == null:
		return
	# VGAMasterNode creates the shared GPU data buffer after our _ready() has
	# already run, so the uniform set can only be built once stepping starts.
	if not _uniform_set.is_valid():
		_create_uniform_set()
	var robot_pos = robot.global_position

	#layout(push_constant, std430) uniform PushConstants {
		#vec2 robot_pos;            8 bytes offset 0
		#float suction_radius;      4 bytes offset 8
		#float collection_radius;   4 bytes offset 12
		#float delta;               4 bytes offset 16
		#uint n_particles;          4 bytes offset 20
		#uint env_index;            4 bytes offset 24
	#} pc;							28 bytes total (have to round to 32 for some reason)
	var push_consts = PackedByteArray()
	push_consts.resize(32) # See above
	push_consts.encode_float(0, robot_pos.x)
	push_consts.encode_float(4, robot_pos.z)
	push_consts.encode_float(8, suction_radius)
	push_consts.encode_float(12, collection_radius)
	push_consts.encode_float(16, delta)
	push_consts.encode_u32(20, n_particles)
	push_consts.encode_u32(24, agent.get_env_index())
	
	var cl = _rd.compute_list_begin()
	
	_rd.compute_list_bind_compute_pipeline(cl, _pipeline)
	_rd.compute_list_bind_uniform_set(cl, _uniform_set, 0)
	_rd.compute_list_set_push_constant(cl, push_consts, push_consts.size())
	_rd.compute_list_dispatch(cl, ceil(n_particles / 64.0), 1, 1)
	
	_rd.compute_list_end()

# Generate dirt particles and upload to GPU
func reset() -> void:
	var data := PackedByteArray()
	data.resize(_byte_count)
	
	# Particles get generated with a Gaussian distribution. 50/50 chance of one vs two centroids:
	var n_clusters := 2 if randf() < 0.5 else 1
	var centers : Array[Vector2] = []
	for _i in n_clusters:
		centers.append(Vector2(
			# centers 0.5 meters away from edge:
			randf_range(-2, 2), randf_range(-2, 2)
		))
	
	for _i in n_particles:
		var center := centers[randi() % n_clusters]
		var x := clampf(randfn(center.x, spread_std), -2.5, 2.5)
		var z := clampf(randfn(center.y, spread_std), -2.5, 2.5)
		
		var offset := _i * 4 * 4 # 4 values, 4 bytes per value
		data.encode_float(offset, x)
		data.encode_float(offset + 4, 0.0)
		data.encode_float(offset + 8, z)
		data.encode_float(offset + 12, 1.0)

		# Add same point to multimesh for rendering. After reset, compute shader will make sure
		# multimesh stays in sync with _buffer.
		_mm.set_instance_transform(_i, Transform3D(
			Basis.IDENTITY, Vector3(x, 0.0001, z)
		))

	_rd.buffer_update(_buffer, 0, data.size(), data)

func _multimesh_stuff() -> void:
	var pm := PlaneMesh.new()
	pm.size = Vector2(0.08, 0.08)
	var mat := ShaderMaterial.new()
	mat.shader = load("res://obs_shader.gdshader")
	mat.set_shader_parameter("dirt_mask", 1.0)
	pm.material = mat
	
	_mm = MultiMesh.new()
	_mm.transform_format = MultiMesh.TRANSFORM_3D
	_mm.mesh = pm
	_mm.instance_count = n_particles
	
	_mmi3D = MultiMeshInstance3D.new()
	_mmi3D.multimesh = _mm
	add_child(_mmi3D)

func _setup_compute() -> void:
	var spriv : RDShaderSPIRV = load("res://dirt_compute.glsl").get_spirv()
	_cp_shader = _rd.shader_create_from_spirv(spriv)
	_pipeline = _rd.compute_pipeline_create(_cp_shader)

func _create_uniform_set() -> void:
	var particles := RDUniform.new()
	particles.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	particles.binding = 0
	particles.add_id(_buffer)
	
	var multi_mesh_buf = RDUniform.new()
	multi_mesh_buf.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	multi_mesh_buf.binding = 1
	multi_mesh_buf.add_id(RenderingServer.multimesh_get_buffer_rd_rid(_mm.get_rid()))
	
	# All envs share this one buffer; pc.env_index picks our slice.
	var vga_data = RDUniform.new()
	vga_data.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	vga_data.binding = 2
	vga_data.add_id(agent.get_gpu_buffer_rid())
	
	_uniform_set = _rd.uniform_set_create(
		[particles, multi_mesh_buf, vga_data],
		_cp_shader,
		0
	)
