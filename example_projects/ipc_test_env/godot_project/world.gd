extends Node3D

# Number of ticks/calls over which we log deltas.
const LOG_COUNT := 10

@onready var _ball: RigidBody3D = $RigidBody3D
@onready var _room_mesh: MeshInstance3D = $Room/CSGBakedMeshInstance3D

var _room_material: StandardMaterial3D
var _tick: int = 0
var _prev_physics_frames: int = -1
var _process_log_count: int = 0

# Colours to cycle through each physics tick so the rendered frame changes visibly.
const COLORS: Array[Color] = [
	Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
	Color.CYAN, Color.MAGENTA, Color.ORANGE, Color.WHITE,
]

func _ready() -> void:
	print("[world] low_processor_usage_mode: ", OS.is_in_low_processor_usage_mode())
	var master := get_parent().get_parent() as VGAMasterNode
	if master:
		print("[world] user_args: ", master.get_user_args())

	_room_material = StandardMaterial3D.new()
	_room_mesh.set_surface_override_material(0, _room_material)

	# _ready() timing probe: set the ball to a random height and observe how
	# many physics ticks pass before it actually lands there. The random height
	# also makes it easy to verify that parallel environments are independently
	# initialized when running via VGAMasterNode.
	var height := randf_range(1.0, 2.5)
	_ball.position = Vector3(0.0, height, 0.0)
	_ball.linear_velocity = Vector3.ZERO
	print("[world] _ready: ball position requested: ", _ball.position)

func _physics_process(delta: float) -> void:
	_tick += 1

	# Assert that the physics frame counter advances by exactly 1 every tick.
	# This will catch catch-up bursts once the engine settings are applied.
	var frames := Engine.get_physics_frames()
	if _prev_physics_frames >= 0:
		var diff := frames - _prev_physics_frames
		if diff != 1:
			push_warning(
				"[world] physics frames jumped by %d (expected 1) at tick %d" % [diff, _tick])
	_prev_physics_frames = frames

	# Log the physics delta for the first LOG_COUNT ticks.
	# Once step_rate_hz is wired to Engine.physics_ticks_per_second, delta
	# should be exactly 1 / step_rate_hz on every tick.
	if _tick <= LOG_COUNT:
		var master := get_parent().get_parent() as VGAMasterNode
		if master:
			var expected := 1.0 / master.step_rate_hz
			print("[world] _physics_process tick=%d  delta=%.9f  expected=%.9f  err=%s"
				% [_tick, delta, expected, str(abs(delta - expected))])
		else:
			print("[world] _physics_process tick=%d  delta=%.9f  (no VGAMasterNode)" % [_tick, delta])

	# _ready() timing probe: log actual ball position for the first few ticks
	# to see when (if ever) it matches what _ready() requested.
	if _tick <= 5:
		print("[world] ball position at tick %d: %s" % [_tick, _ball.position])

	# Set colour from the action so Python can verify visual obs match what was sent.
	var agent := $VGAAgentNode as VGAAgentNode
	_room_material.albedo_color = COLORS[agent.last_action_byte % COLORS.size()]

func _process(delta: float) -> void:
	# Log the first LOG_COUNT _process deltas. After the engine settings land,
	# time_scale = 1e6 will make these astronomically large — confirming that
	# _process is not suitable for simulation logic.
	if _process_log_count < LOG_COUNT:
		_process_log_count += 1
		print("[world] _process call=%d  delta=%.6f" % [_process_log_count, delta])
