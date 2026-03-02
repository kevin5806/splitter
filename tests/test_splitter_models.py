import unittest
from pathlib import Path

from src.splitter_models import ImageItem, ImageSplitterConfig


class ImageItemModelTests(unittest.TestCase):
    def test_post_init_normalizes_path(self):
        item = ImageItem("C:/tmp/example.png")
        self.assertIsInstance(item.file_path, Path)
        self.assertEqual(item.file_path.name, "example.png")

    def test_default_item_uses_global_defaults_summary(self):
        item = ImageItem("sample.jpg")
        self.assertFalse(item.has_custom_settings())
        self.assertEqual(item.get_settings_summary(), "Using global defaults")
        self.assertEqual(item.get_display_name(), "sample.jpg")

    def test_crop_margin_alone_marks_custom_and_summary(self):
        item = ImageItem("sample.jpg", crop_margin=8)
        self.assertTrue(item.has_custom_settings())
        self.assertIn("Margin: 8px", item.get_settings_summary())
        self.assertIn("sample.jpg", item.get_display_name())
        self.assertNotEqual(item.get_display_name(), "sample.jpg")

    def test_orientation_alone_marks_custom_and_summary(self):
        item = ImageItem("sample.jpg", orientation_mode="vertical")
        self.assertTrue(item.has_custom_settings())
        self.assertIn("Orientation: Vertical", item.get_settings_summary())

    def test_whitespace_folder_is_not_treated_as_custom(self):
        item = ImageItem("sample.jpg", custom_folder="   ")
        self.assertFalse(item.has_custom_settings())
        self.assertEqual(item.get_settings_summary(), "Using global defaults")

    def test_summary_contains_all_overrides(self):
        item = ImageItem(
            "sample.jpg",
            base_size=1024,
            images_across=3,
            images_high=4,
            maintain_format=True,
            smart_grid=False,
            crop_margin=6,
            orientation_mode="horizontal",
            custom_folder="pack_a",
        )
        summary = item.get_settings_summary()
        self.assertIn("Size: 1024", summary)
        self.assertIn("Grid: 3x4", summary)
        self.assertIn("Keep format", summary)
        self.assertIn("Classic grid", summary)
        self.assertIn("Margin: 6px", summary)
        self.assertIn("Orientation: Horizontal", summary)
        self.assertIn("Folder: pack_a", summary)


class ImageSplitterConfigModelTests(unittest.TestCase):
    def test_default_config_state(self):
        cfg = ImageSplitterConfig()
        self.assertEqual(cfg.image_items, [])
        self.assertIsNone(cfg.selected_item)
        self.assertFalse(cfg.processing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
