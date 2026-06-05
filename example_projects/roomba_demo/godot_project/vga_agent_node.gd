extends VGAAgentNode

@onready var robot: RoombaRobot = get_parent()

# Steps before episode gets truncated:
@export var total_steps : int = 1500
var episode_steps : int = 0

var forward : float = 0.0
var strafe : float = 0.0
var turn : float = 0.0

var dirt_sys: DirtSystem
var trail_sys: TrailSystem
var collected_dirt: int = 0

func _reset() -> void:
	episode_steps = 0
	forward = 0.0
	strafe = 0.0
	turn = 0.0
	collected_dirt = 0
	robot.reset()
	if dirt_sys != null:
		dirt_sys.reset()
	if trail_sys != null:
		trail_sys.reset()

func _apply_action(p_action: PackedByteArray) -> void:
	forward = clampf(p_action.decode_float(0), -1.0, 1.0)
	strafe = clampf(p_action.decode_float(4), -1.0, 1.0)
	turn = clampf(p_action.decode_float(8), -1.0, 1.0)

func _collect_scalar_obs() -> PackedByteArray:
	return PackedByteArray([1])

func _get_reward() -> float:
	# NOTE: doing it in this very inefficient manner currently. Later on want to do it in a more
	# efficient way that doesn't require N buffer_get_data calls.
	if dirt_sys == null:
		return 0.0
	var collected_dirt_latest = dirt_sys.get_total_collected()
	var dirt_reward = (collected_dirt_latest - collected_dirt) / float(dirt_sys.n_particles)
	collected_dirt = collected_dirt_latest
	
	# Penalize if robot gets beyond border of the map.
	var fall_off_penalty := 0.0
	if abs(robot.global_position.x) > 2.5 or abs(robot.global_position.z) > 2.5:
		fall_off_penalty = -1.0
	
	# To make it hurry up more:
	var step_penalty := -0.001
	
	return dirt_reward + fall_off_penalty + step_penalty

func _get_episode_state() -> int:
	episode_steps += 1
	if dirt_sys == null:
		return RUNNING
	if collected_dirt >= dirt_sys.n_particles:
		return TERMINATED
	if abs(robot.global_position.x) > 2.5 or abs(robot.global_position.z) > 2.5:
		return TERMINATED
	if episode_steps >= total_steps:
		return TRUNCATED
	return RUNNING

func _get_action_size() -> int:
	# 3 floats: forward, strafe and turn (4 bytes per float)
	return 3 * 4

func _get_scalar_obs_size() -> int:
	return 1
