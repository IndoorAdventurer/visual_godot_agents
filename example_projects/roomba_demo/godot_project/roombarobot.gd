class_name RoombaRobot
extends CharacterBody3D

const SPEED := 2.0
const TURN_SPEED := 2.0

@onready var agent := $VGAAgentNode
var _human_mode := true

var _spawn_pos: Vector3
var _just_reset: bool = false

func _ready() -> void:
	_human_mode = get_tree().root.get_node_or_null("VGAMasterNode") == null
	_spawn_pos = global_position

func _physics_process(delta: float) -> void:
	var strafe: float; var forward: float; var turn: float
	if _human_mode:
		strafe = Input.get_axis("move_left", "move_right")
		forward = Input.get_axis("move_backward", "move_forward")
		turn = Input.get_axis("turn_left", "turn_right")
		
		# Manually doing the reward and reset functionality from here
		agent._get_reward()
		if agent._get_episode_state() != agent.RUNNING:
			agent._reset()
	else:
		strafe = agent.strafe
		forward = agent.forward
		turn = agent.turn
	
	var input_dir := Vector3(strafe, 0.0, -forward)
	if input_dir.length() > 1.0:
		input_dir = input_dir.normalized()
	
	rotate_y(-turn * TURN_SPEED * delta)
	velocity = transform.basis * input_dir * SPEED
	
	if not _just_reset:
		move_and_slide()
	_just_reset = false

func reset() -> void:
	_just_reset = true
	global_position = _spawn_pos
	velocity = Vector3.ZERO
