extends Node3D

@onready var robot := $Robot
@onready var agent := $Robot/HPAAgentNode
@onready var dirt_sys := $World/DirtSystem
@onready var trail_sys := $World/StaticBody3D/MeshInstance3D

func _ready() -> void:
	dirt_sys.robot = robot
	trail_sys.robot = robot
	agent.dirt_sys = dirt_sys
	agent.trail_sys = trail_sys
