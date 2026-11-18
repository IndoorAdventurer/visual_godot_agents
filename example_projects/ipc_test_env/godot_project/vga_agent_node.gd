extends VGAAgentNode

# Last action byte received; read by world.gd to select the room colour.
var last_action_byte: int = 0

# Incremented each time _reset() is called; echoed in scalar obs so Python
# can verify that reset was invoked exactly when expected.
var reset_count: int = 0

# Messages received so far, the last one, and the count when _reset() last ran.
var message_count: int = 0
var last_message := PackedByteArray([0, 0, 0, 0])
var message_count_at_reset: int = -1

# _physics_process calls so far.
var physics_ticks: int = 0


func _get_scalar_obs_size() -> int:
	# Three float32s: [last_action_byte + 10.0, reset_count, episode_state_echo]
	# episode_state_echo: 0.0 = RUNNING, 254.0 = TRUNCATED, 255.0 = TERMINATED
	return 12

func _get_action_size() -> int:
	return 1

func _get_message_size() -> int:
	return 4

func _apply_message(message: PackedByteArray) -> void:
	message_count += 1
	last_message = message

func _get_info_size() -> int:
	# int32s: [global index, local index, total envs, reset_count, message_count,
	# message_count_at_reset], the last message's 4 bytes, then int32
	# physics_ticks.
	return 32

func _collect_info() -> PackedByteArray:
	var master := get_tree().current_scene as VGAMasterNode
	var buf := PackedByteArray()
	buf.resize(32)
	buf.encode_s32(0, get_global_env_index())
	buf.encode_s32(4, get_local_env_index())
	buf.encode_s32(8, master.get_total_num_envs() if master else -1)
	buf.encode_s32(12, reset_count)
	buf.encode_s32(16, message_count)
	buf.encode_s32(20, message_count_at_reset)
	for i in 4:
		buf[24 + i] = last_message[i]
	buf.encode_s32(28, physics_ticks)
	return buf

func _collect_scalar_obs() -> PackedByteArray:
	var state := _get_episode_state()
	var state_echo := 255.0 if state == TERMINATED else (254.0 if state == TRUNCATED else 0.0)
	var buf := PackedByteArray()
	buf.resize(12)
	buf.encode_float(0, float(last_action_byte) + 10.0)
	buf.encode_float(4, float(reset_count))
	buf.encode_float(8, state_echo)
	return buf

func _get_reward() -> float:
	return randf() * 5.0

func _get_episode_state() -> int:
	# Sentinel actions trigger episode end so the auto-reset path can be tested.
	# 255 → natural termination, 254 → truncation. All other actions → running.
	if last_action_byte == 255:
		return TERMINATED
	if last_action_byte == 254:
		return TRUNCATED
	return RUNNING

func _physics_process(_delta: float) -> void:
	physics_ticks += 1

func _reset() -> void:
	reset_count += 1
	message_count_at_reset = message_count
	last_action_byte = 0

func _apply_action(action: PackedByteArray) -> void:
	if action.size() > 0:
		last_action_byte = action[0]
