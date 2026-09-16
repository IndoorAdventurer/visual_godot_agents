"""
Pieces shared by the frame-stacking wrappers.

Kept in their own module so naive_frame_stack does not pull in torch, which
indexed_frame_stack requires.
"""

import gymnasium

PADDING_MODES = ("repeat", "zero")
SINGLE_KEY = None  # store key used when the wrapped space is a plain Box


def validate_stacking(stack: int, stride: int, padding: str) -> None:
    """Raise ValueError if any stacking option is out of range."""
    if stack < 1:
        raise ValueError(f"stack must be >= 1, got {stack}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if padding not in PADDING_MODES:
        raise ValueError(f"padding must be one of {PADDING_MODES}, got {padding!r}")


def leaf_spaces(space) -> dict:
    """Observation space flattened into the leaves that get stacked."""
    if isinstance(space, gymnasium.spaces.Box):
        return {SINGLE_KEY: space}
    if isinstance(space, gymnasium.spaces.Dict):
        for key, leaf in space.spaces.items():
            if not isinstance(leaf, gymnasium.spaces.Box):
                raise TypeError(f"observation leaf {key!r} must be a Box, got {type(leaf).__name__}")
        return dict(space.spaces)
    raise TypeError(f"observation space must be Box or Dict of Boxes, got {type(space).__name__}")
