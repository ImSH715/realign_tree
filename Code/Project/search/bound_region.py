from dataclasses import dataclass


@dataclass
class BoundedRegion:
    center_x: float
    center_y: float
    radius_px: int


def build_bounded_region(center_x: float, center_y: float, radius_px: int) -> BoundedRegion:
    return BoundedRegion(center_x=center_x, center_y=center_y, radius_px=radius_px)