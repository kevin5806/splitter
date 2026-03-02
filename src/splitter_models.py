from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class ImageItem:
    """Represents an image with its individual processing settings."""

    file_path: Path
    base_size: Optional[int] = None
    custom_size: Optional[int] = None
    images_across: Optional[int] = None
    images_high: Optional[int] = None
    maintain_format: Optional[bool] = None
    smart_grid: Optional[bool] = None
    crop_margin: Optional[int] = None
    custom_folder: Optional[str] = None

    def __post_init__(self):
        self.file_path = Path(self.file_path)

    def get_display_name(self):
        name = self.file_path.name
        if self.has_custom_settings():
            name = f"⚙ {name}"
        return name

    def has_custom_settings(self):
        has_custom_folder = bool(self.custom_folder and str(self.custom_folder).strip())
        return any(
            [
                self.base_size is not None,
                self.custom_size is not None,
                self.images_across is not None,
                self.images_high is not None,
                self.maintain_format is not None,
                self.smart_grid is not None,
                self.crop_margin is not None,
                has_custom_folder,
            ]
        )

    def get_settings_summary(self):
        if not self.has_custom_settings():
            return "Using global defaults"

        parts = []
        if self.custom_size:
            parts.append(f"Size: {self.custom_size}")
        elif self.base_size:
            parts.append(f"Size: {self.base_size}")

        if self.images_across or self.images_high:
            across = self.images_across or 1
            high = self.images_high or 1
            parts.append(f"Grid: {across}x{high}")

        if self.maintain_format is not None:
            parts.append("Keep format" if self.maintain_format else "Convert to JPEG")

        if self.smart_grid is not None:
            parts.append("Smart grid" if self.smart_grid else "Classic grid")

        if self.crop_margin is not None:
            parts.append(f"Margin: {self.crop_margin}px")

        if self.custom_folder and self.custom_folder.strip():
            parts.append(f"Folder: {self.custom_folder}")

        return " | ".join(parts) if parts else "Using global defaults"


@dataclass
class ImageSplitterConfig:
    """Configuration and runtime state for image splitting operations."""

    image_items: List[ImageItem] = field(default_factory=list)
    selected_item: Optional[ImageItem] = None
    processing: bool = False
