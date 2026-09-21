class_name VGASpectator
extends SubViewport
## A human view of the environment it is placed in, shown on the main window.
##
## Add it to the environment scene with a [Camera3D] child. It is shown when the
## [VGAMasterNode] runs in [member VGAMasterNode.real_time_mode] with a single
## environment, and frees itself otherwise, so check outside references to it with
## [method @GlobalScope.is_instance_valid].
##
## Input on the main window reaches [method Node._input] below it. Move the camera in
## [method Node._physics_process]: [method Node._process]'s delta is unreliable in
## real-time mode.

var _display: TextureRect


func _ready() -> void:
	# Assumes the master is the main scene's root.
	var master := get_tree().current_scene as VGAMasterNode
	if master == null or not master.real_time_mode:
		queue_free()	# Delete self if not used.
		return
	if master.num_envs != 1:
		push_warning("VGASpectator: only shown when num_envs is 1, got %d." % master.num_envs)
		queue_free()
		return

	var window := get_window()
	size = window.size
	window.size_changed.connect(_on_window_size_changed)
	# A nested SubViewport receives no input on its own.
	window.window_input.connect(push_input)

	_display = TextureRect.new()
	_display.texture = get_texture()
	_display.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_display.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	# The root is still readying the master when this runs.
	window.add_child.call_deferred(_display)


func _exit_tree() -> void:
	if _display != null:
		_display.queue_free()
		_display = null


func _on_window_size_changed() -> void:
	size = get_window().size
