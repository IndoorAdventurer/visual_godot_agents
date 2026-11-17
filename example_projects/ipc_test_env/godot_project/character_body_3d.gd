extends CharacterBody3D

@export var movement_speed := 5.0
@export var rotation_speed := 5.0
@onready var camera = $Camera3D

func _ready() -> void:
	position.x += randf_range(-1.0, 1.0)
	position.z += randf_range(-1.0, 1.0)
	rotate_y(randf_range(-PI, PI))

func _physics_process(delta: float) -> void:
	# Get inputs:
	var input_dir := Input.get_vector("move_left", "move_right", "move_forward", "move_backward")
	var input_rot := Input.get_vector("look_up", "look_down", "turn_left", "turn_right")
	
	# Rotation and movement:
	rotation.y -= input_rot.y * delta * rotation_speed
	camera.rotation.x -= input_rot.x * delta * rotation_speed
	camera.rotation.x = clampf(camera.rotation.x, -1.3, 1.3)
	velocity = transform.basis *  Vector3(input_dir.x, 0, input_dir.y) * movement_speed

	# Actually making it move properly in the world:
	_push_stuff_away()
	move_and_slide()

## Makes sure there are some physics interactions with RigidBody objects
func _push_stuff_away():
	for idx in get_slide_collision_count():
		var collision := get_slide_collision(idx)
		var collider:= collision.get_collider()
		if collider is RigidBody3D:
			var push_dir := collision.get_normal()
			collider.apply_central_impulse(-push_dir * 10)
