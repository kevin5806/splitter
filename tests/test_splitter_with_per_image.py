import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import splitter_with_per_image as splitter


def _make_grid_image(path, cols, rows, width=480, height=480, image_format="JPEG"):
    """Create a synthetic sheet with white separators and dark cells."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin = 20
    gap = 12
    cell_w = (width - (2 * margin) - ((cols - 1) * gap)) // cols
    cell_h = (height - (2 * margin) - ((rows - 1) * gap)) // rows

    for row in range(rows):
        for col in range(cols):
            left = margin + col * (cell_w + gap)
            top = margin + row * (cell_h + gap)
            right = left + cell_w
            bottom = top + cell_h
            draw.rectangle((left, top, right, bottom), fill=(20, 20, 20))

    img.save(path, format=image_format)
    return path


class SplitterUnitTests(unittest.TestCase):
    def test_index_to_alpha(self):
        self.assertEqual(splitter._index_to_alpha(0), "A")
        self.assertEqual(splitter._index_to_alpha(25), "Z")
        self.assertEqual(splitter._index_to_alpha(26), "AA")
        self.assertEqual(splitter._index_to_alpha(27), "AB")
        self.assertEqual(splitter._index_to_alpha(52), "BA")

    def test_generate_crop_codes(self):
        codes = splitter._generate_crop_codes(4, 4, 16)
        self.assertEqual(len(codes), 16)
        self.assertEqual(codes[0], "A1")
        self.assertEqual(codes[3], "A4")
        self.assertEqual(codes[4], "B1")
        self.assertEqual(codes[-1], "D4")

        fallback_codes = splitter._generate_crop_codes(4, 4, 3)
        self.assertEqual(fallback_codes, ["P1", "P2", "P3"])

    def test_create_output_folder_with_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            image_path = source_dir / "sample.jpg"
            _make_grid_image(image_path, cols=2, rows=2)

            output_dir = tmp_path / "exports"
            out = splitter.create_output_folder(
                image_path,
                custom_folder="album",
                images_across=4,
                images_high=4,
                timestamp="20260302_101010",
                output_base_dir=output_dir,
            )
            self.assertEqual(out, output_dir / "album" / "sample")
            self.assertTrue(out.exists())

    def test_split_and_resize_generates_tracking_files_and_short_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.jpg"
            _make_grid_image(image_path, cols=2, rows=2, image_format="JPEG")

            progress = []

            def progress_cb(count, total):
                progress.append((count, total))

            output_folder = splitter.split_and_resize_image(
                image_path=str(image_path),
                images_across=2,
                images_high=2,
                output_size=64,
                custom_folder="cuts",
                maintain_format=False,
                smart_grid=False,
                timestamp="20260302_121314",
                output_base_dir=tmp_path / "output-root",
                progress_callback=progress_cb,
            )

            self.assertEqual(output_folder, tmp_path / "output-root" / "cuts" / "sheet")
            self.assertTrue(output_folder.exists())

            expected_parts = {"A1.jpg", "A2.jpg", "B1.jpg", "B2.jpg"}
            produced_files = {p.name for p in output_folder.iterdir() if p.is_file()}
            self.assertTrue(expected_parts.issubset(produced_files))
            self.assertIn("sheet_tracking_overlay.jpg", produced_files)
            self.assertIn("sheet_tracking_index.csv", produced_files)
            self.assertIn("sheet_diagnostics.json", produced_files)

            self.assertEqual(progress[-1], (4, 4))

            csv_path = output_folder / "sheet_tracking_index.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["code"], "A1")
            self.assertEqual(rows[0]["filename"], "A1.jpg")

            diagnostics_path = output_folder / "sheet_diagnostics.json"
            with diagnostics_path.open("r", encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            self.assertIn("confidence_score", diagnostics)
            self.assertIn("method", diagnostics)
            self.assertEqual(diagnostics["total_parts"], 4)

    def test_split_and_resize_maintain_png_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"
            _make_grid_image(image_path, cols=1, rows=1, image_format="PNG")

            output_folder = splitter.split_and_resize_image(
                image_path=str(image_path),
                images_across=1,
                images_high=1,
                output_size=64,
                custom_folder="cuts",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_222222",
                output_base_dir=tmp_path / "out",
            )
            produced = {p.name for p in output_folder.iterdir() if p.is_file()}
            self.assertIn("A1.png", produced)

            csv_path = output_folder / "sheet_tracking_index.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["filename"], "A1.png")

    def test_split_and_resize_applies_crop_margin_and_tracks_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "single.jpg"

            img = Image.new("RGB", (80, 80), "black")
            img.save(image_path, format="JPEG")

            output_folder = splitter.split_and_resize_image(
                image_path=str(image_path),
                images_across=1,
                images_high=1,
                output_size=100,
                custom_folder="cuts",
                maintain_format=False,
                smart_grid=False,
                timestamp="20260302_333333",
                output_base_dir=tmp_path / "out",
                crop_margin=10,
            )

            out_img_path = output_folder / "A1.jpg"
            with Image.open(out_img_path) as out_img:
                self.assertEqual(out_img.size, (100, 100))
                corner = out_img.getpixel((1, 1))
                center = out_img.getpixel((50, 50))
                self.assertTrue(all(channel >= 240 for channel in corner))
                self.assertTrue(all(channel <= 30 for channel in center))

            csv_path = output_folder / "single_tracking_index.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["left"], "-10")
            self.assertEqual(rows[0]["top"], "-10")
            self.assertEqual(rows[0]["right"], "90")
            self.assertEqual(rows[0]["bottom"], "90")

    @unittest.skipUnless(splitter.CV_AVAILABLE, "OpenCV not available")
    def test_detect_smart_grid_boxes_on_clean_synthetic_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "clean_grid.jpg"
            _make_grid_image(image_path, cols=4, rows=4, width=800, height=800, image_format="JPEG")
            with Image.open(image_path) as img:
                boxes = splitter.detect_smart_grid_boxes(img, images_across=4, images_high=4)

            self.assertIsNotNone(boxes)
            self.assertEqual(len(boxes), 16)
            for left, top, right, bottom in boxes:
                self.assertGreater(right, left)
                self.assertGreater(bottom, top)

    def test_crop_with_white_padding_for_out_of_bounds_box(self):
        img = Image.new("RGB", (10, 10), (0, 0, 0))
        padded = splitter._crop_with_white_padding(img, (-2, -3, 5, 4))
        self.assertEqual(padded.size, (7, 7))
        self.assertEqual(padded.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(padded.getpixel((3, 3)), (0, 0, 0))

    def test_assess_image_croppability_flags_impossible_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "white.jpg"
            Image.new("RGB", (800, 800), "white").save(image_path, format="JPEG")

            is_ok, reason, diagnostics = splitter.assess_image_croppability(
                image_path=str(image_path),
                images_across=4,
                images_high=4,
                smart_grid=True,
                min_confidence=12.0,
            )

            self.assertFalse(is_ok)
            self.assertIn(reason, {"grid_not_detected", "too_few_detected_lines", "very_low_confidence"})

    def test_extract_accuracy_score_handles_invalid_and_clamps(self):
        self.assertIsNone(splitter.extract_accuracy_score(None))
        self.assertIsNone(splitter.extract_accuracy_score({"confidence_score": "abc"}))
        self.assertEqual(splitter.extract_accuracy_score({"confidence_score": -5}), 0.0)
        self.assertEqual(splitter.extract_accuracy_score({"confidence_score": 112.5}), 100.0)
        self.assertEqual(splitter.extract_accuracy_score({"confidence_score": 74.25}), 74.25)

    def test_format_accuracy_score_and_band(self):
        self.assertEqual(splitter.classify_accuracy_band(None), "n/a")
        self.assertEqual(splitter.classify_accuracy_band(89), "Alta")
        self.assertEqual(splitter.classify_accuracy_band(70), "Media")
        self.assertEqual(splitter.classify_accuracy_band(40), "Bassa")
        self.assertEqual(splitter.format_accuracy_score(None), "-")
        self.assertEqual(splitter.format_accuracy_score(70), "70.0 (Media)")

    def test_build_diagnostic_summary_contains_key_signals(self):
        diagnostics = {
            "method": "direct_detected",
            "orientation_mode": "horizontal",
            "malformed_cell_flags": 2,
            "offscreen_cells": 1,
            "white_border_trimmed_crops": 3,
            "overlap_corrected_pairs": 4,
            "remaining_overlap_pairs": 0,
            "inferred_vertical_separators": 1,
            "inferred_horizontal_separators": 2,
        }
        summary = splitter.build_diagnostic_summary(diagnostics)
        self.assertIn("method=direct_detected", summary)
        self.assertIn("orientation=horizontal", summary)
        self.assertIn("malformed=2", summary)
        self.assertIn("offscreen=1", summary)
        self.assertIn("white_trim=3", summary)
        self.assertIn("overlap_fixed=4", summary)
        self.assertIn("inferred=v1/h2", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
