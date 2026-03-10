"""Motion graphics tools — text overlay and template parameter computation.

Pure logic layer: no GES dependency. Computes parameters that the GES
execution layer applies to the timeline.
"""

from dataclasses import dataclass
from enum import Enum


class MotionGraphicsError(Exception):
    """Raised when motion graphics parameter validation fails."""


class TextPosition(Enum):
    """Supported text positions within a frame."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


# Font size limits
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 500


@dataclass(frozen=True)
class TextOverlayParams:
    """Computed text overlay parameters."""

    text: str
    font_family: str
    font_size: int
    position: TextPosition
    color: tuple[int, int, int, int]
    duration_ns: int
    bg_color: tuple[int, int, int, int] | None
    padding: int


@dataclass(frozen=True)
class LowerThirdParams:
    """Computed lower third template parameters."""

    name_params: TextOverlayParams
    title_params: TextOverlayParams
    bg_rect: tuple[int, int, int, int]  # x, y, width, height
    duration_ns: int


def _validate_color(color: tuple[int, int, int, int], label: str) -> None:
    """Validate an RGBA color tuple."""
    if len(color) != 4:
        raise MotionGraphicsError(f"Color must have 4 components (RGBA), got {len(color)}")
    for i, c in enumerate(color):
        if c < 0 or c > 255:
            raise MotionGraphicsError(
                f"Color component {i} of {label} must be 0-255, got {c}"
            )


def compute_text_overlay(
    text: str,
    font_family: str,
    font_size: int,
    position: TextPosition,
    color: tuple[int, int, int, int],
    duration_ns: int,
    bg_color: tuple[int, int, int, int] | None = None,
    padding: int = 0,
) -> TextOverlayParams:
    """Validate and compute text overlay parameters.

    Args:
        text: Non-empty text string.
        font_family: Font family name (e.g. "Arial").
        font_size: Font size in points (8-500).
        position: TextPosition enum value.
        color: RGBA tuple (0-255 each).
        duration_ns: Duration in nanoseconds (positive).
        bg_color: Optional RGBA background color tuple.
        padding: Padding in pixels (0-200).

    Returns:
        TextOverlayParams frozen dataclass.

    Raises:
        MotionGraphicsError: If parameters are invalid.
    """
    if not text:
        raise MotionGraphicsError("Text must not be empty")

    if font_size < MIN_FONT_SIZE or font_size > MAX_FONT_SIZE:
        raise MotionGraphicsError(
            f"Font size must be {MIN_FONT_SIZE}-{MAX_FONT_SIZE}, got {font_size}"
        )

    if duration_ns <= 0:
        raise MotionGraphicsError(f"Duration must be positive, got {duration_ns}")

    _validate_color(color, "color")

    if bg_color is not None:
        _validate_color(bg_color, "bg_color")

    if padding < 0 or padding > 200:
        raise MotionGraphicsError(f"Padding must be 0-200, got {padding}")

    return TextOverlayParams(
        text=text,
        font_family=font_family,
        font_size=font_size,
        position=position,
        color=color,
        duration_ns=duration_ns,
        bg_color=bg_color,
        padding=padding,
    )


def compute_position_coords(
    position: TextPosition,
    frame_width: int,
    frame_height: int,
    text_width: int,
    text_height: int,
    padding: int,
) -> tuple[int, int]:
    """Convert a TextPosition enum to pixel x,y coordinates.

    Args:
        position: TextPosition enum value.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        text_width: Text element width in pixels.
        text_height: Text element height in pixels.
        padding: Padding from edges in pixels.

    Returns:
        Tuple of (x, y) pixel coordinates.
    """
    # Horizontal positions
    left_x = padding
    center_x = (frame_width - text_width) // 2
    right_x = frame_width - text_width - padding

    # Vertical positions
    top_y = padding
    center_y = (frame_height - text_height) // 2
    bottom_y = frame_height - text_height - padding

    _POSITION_MAP = {
        TextPosition.TOP_LEFT: (left_x, top_y),
        TextPosition.TOP_CENTER: (center_x, top_y),
        TextPosition.TOP_RIGHT: (right_x, top_y),
        TextPosition.CENTER: (center_x, center_y),
        TextPosition.BOTTOM_LEFT: (left_x, bottom_y),
        TextPosition.BOTTOM_CENTER: (center_x, bottom_y),
        TextPosition.BOTTOM_RIGHT: (right_x, bottom_y),
    }

    return _POSITION_MAP[position]


def compute_lower_third(
    name: str,
    title: str,
    frame_width: int,
    frame_height: int,
    duration_ns: int,
    font_family: str = "Arial",
) -> LowerThirdParams:
    """Compute parameters for a lower third template.

    Template: name in larger font, title in smaller font, semi-transparent
    background bar at bottom of frame.

    Args:
        name: Person's name (non-empty).
        title: Person's title/role (non-empty).
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        duration_ns: Duration in nanoseconds.
        font_family: Font family name.

    Returns:
        LowerThirdParams frozen dataclass.

    Raises:
        MotionGraphicsError: If parameters are invalid.
    """
    if not name:
        raise MotionGraphicsError("Name must not be empty")
    if not title:
        raise MotionGraphicsError("Title must not be empty")
    if duration_ns <= 0:
        raise MotionGraphicsError(f"Duration must be positive, got {duration_ns}")

    name_font_size = 36
    title_font_size = 24
    bg_height = 120
    padding = 40

    name_params = TextOverlayParams(
        text=name,
        font_family=font_family,
        font_size=name_font_size,
        position=TextPosition.BOTTOM_LEFT,
        color=(255, 255, 255, 255),
        duration_ns=duration_ns,
        bg_color=None,
        padding=padding,
    )

    title_params = TextOverlayParams(
        text=title,
        font_family=font_family,
        font_size=title_font_size,
        position=TextPosition.BOTTOM_LEFT,
        color=(200, 200, 200, 255),
        duration_ns=duration_ns,
        bg_color=None,
        padding=padding,
    )

    bg_rect = (0, frame_height - bg_height, frame_width, bg_height)

    return LowerThirdParams(
        name_params=name_params,
        title_params=title_params,
        bg_rect=bg_rect,
        duration_ns=duration_ns,
    )


def compute_title_card(
    text: str,
    frame_width: int,
    frame_height: int,
    duration_ns: int,
    font_family: str = "Arial",
    font_size: int = 72,
) -> TextOverlayParams:
    """Compute parameters for a centered title card.

    Args:
        text: Title text (non-empty).
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        duration_ns: Duration in nanoseconds.
        font_family: Font family name.
        font_size: Font size in points.

    Returns:
        TextOverlayParams frozen dataclass.

    Raises:
        MotionGraphicsError: If parameters are invalid.
    """
    return compute_text_overlay(
        text=text,
        font_family=font_family,
        font_size=font_size,
        position=TextPosition.CENTER,
        color=(255, 255, 255, 255),
        duration_ns=duration_ns,
    )
