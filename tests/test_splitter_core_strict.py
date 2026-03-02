import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from src import splitter_core as core


def _save_solid_image(path, size=(80, 60), color=(0, 0, 0), image_format="PNG"):
    image = Image.new("RGB", size, color)
    image.save(path, format=image_format)
    return path


def _make_rotated_content_image(size=(220, 220), rotation_deg=7.0):
    rect = Image.new("RGB", size, "white")
    # Draw an axis-aligned dark rectangle, then rotate full image to simulate skew.
    from PIL import ImageDraw

    rect_draw = ImageDraw.Draw(rect)
    rect_draw.rectangle((55, 85, 165, 135), fill=(0, 0, 0))
    return rect.rotate(rotation_deg, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))


def _make_irregular_grid_image(path, col_widths, row_heights, gap=12, margin=20, image_format="JPEG"):
    width = (2 * margin) + sum(col_widths) + (len(col_widths) - 1) * gap
    height = (2 * margin) + sum(row_heights) + (len(row_heights) - 1) * gap

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = margin
    for row_h in row_heights:
        x = margin
        for col_w in col_widths:
            draw.rectangle((x, y, x + col_w, y + row_h), fill=(20, 20, 20))
            x += col_w + gap
        y += row_h + gap

    img.save(path, format=image_format)
    return path


def _read_tracking_rows(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_diagnostics(json_path):
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class SplitterCoreStrictTests(unittest.TestCase):
    def test_generate_uniform_grid_boxes_exact_geometry(self):
        boxes = core.generate_uniform_grid_boxes(100, 80, images_across=3, images_high=2)
        expected = [
            (0, 0, 33, 40),
            (33, 0, 67, 40),
            (67, 0, 100, 40),
            (0, 40, 33, 80),
            (33, 40, 67, 80),
            (67, 40, 100, 80),
        ]
        self.assertEqual(boxes, expected)

    def test_expand_box_with_margin_supports_negative_zero_and_positive(self):
        box = (1, 2, 3, 4)
        self.assertEqual(core._expand_box_with_margin(box, -5), box)
        self.assertEqual(core._expand_box_with_margin(box, 0), box)
        self.assertEqual(core._expand_box_with_margin(box, 2), (-1, 0, 5, 6))

    def test_white_fill_values_per_mode(self):
        self.assertEqual(core._white_fill_for_mode("L"), 255)
        self.assertEqual(core._white_fill_for_mode("LA"), (255, 255))
        self.assertEqual(core._white_fill_for_mode("RGBA"), (255, 255, 255, 255))
        self.assertEqual(core._white_fill_for_mode("CMYK"), (0, 0, 0, 0))
        self.assertEqual(core._white_fill_for_mode("RGB"), (255, 255, 255))

    def test_create_output_folder_single_cell_uses_timestamp_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "source.png"
            _save_solid_image(image_path, image_format="PNG")

            out_custom = core.create_output_folder(
                image_path=image_path,
                custom_folder="album",
                images_across=1,
                images_high=1,
                timestamp="20260302_121212",
                output_base_dir=tmp_path / "exports",
            )
            self.assertEqual(out_custom, tmp_path / "exports" / "album" / "20260302_121212")
            self.assertTrue(out_custom.exists())

            out_default = core.create_output_folder(
                image_path=image_path,
                custom_folder="",
                images_across=1,
                images_high=1,
                timestamp="20260302_131313",
                output_base_dir=tmp_path / "exports",
            )
            self.assertEqual(out_default, tmp_path / "exports" / "20260302_131313")
            self.assertTrue(out_default.exists())

    def test_split_and_resize_uniform_exact_tracking_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"
            _save_solid_image(image_path, size=(80, 60), image_format="PNG")

            out = core.split_and_resize_image(
                image_path=str(image_path),
                images_across=2,
                images_high=2,
                output_size=40,
                custom_folder="cuts",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_141414",
                output_base_dir=tmp_path / "out",
                crop_margin=0,
            )

            rows = _read_tracking_rows(out / "sheet_tracking_index.csv")
            self.assertEqual(rows, [
                {"code": "A1", "filename": "A1.png", "left": "0", "top": "0", "right": "40", "bottom": "30", "width": "40", "height": "30"},
                {"code": "A2", "filename": "A2.png", "left": "40", "top": "0", "right": "80", "bottom": "30", "width": "40", "height": "30"},
                {"code": "B1", "filename": "B1.png", "left": "0", "top": "30", "right": "40", "bottom": "60", "width": "40", "height": "30"},
                {"code": "B2", "filename": "B2.png", "left": "40", "top": "30", "right": "80", "bottom": "60", "width": "40", "height": "30"},
            ])

            diagnostics = _read_diagnostics(out / "sheet_diagnostics.json")
            self.assertEqual(diagnostics["crop_margin_px"], 0)
            self.assertEqual(diagnostics["total_parts"], 4)
            self.assertEqual(diagnostics["method"], "uniform_fallback")
            self.assertIn("auto_straightened_crops", diagnostics)
            self.assertIn("max_abs_straightening_deg", diagnostics)
            self.assertIn("white_border_trimmed_crops", diagnostics)
            self.assertIn("overlap_guard_active", diagnostics)

    def test_split_and_resize_negative_margin_is_clamped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"
            _save_solid_image(image_path, size=(80, 60), image_format="PNG")

            out = core.split_and_resize_image(
                image_path=str(image_path),
                images_across=2,
                images_high=2,
                output_size=40,
                custom_folder="cuts_neg",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_151515",
                output_base_dir=tmp_path / "out",
                crop_margin=-12,
            )

            rows = _read_tracking_rows(out / "sheet_tracking_index.csv")
            self.assertEqual(rows[0]["left"], "0")
            self.assertEqual(rows[0]["top"], "0")
            self.assertEqual(rows[0]["right"], "40")
            self.assertEqual(rows[0]["bottom"], "30")

            diagnostics = _read_diagnostics(out / "sheet_diagnostics.json")
            self.assertEqual(diagnostics["crop_margin_px"], 0)

    def test_split_and_resize_positive_margin_expands_boxes_and_pads_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"
            _save_solid_image(image_path, size=(80, 60), image_format="PNG")

            out = core.split_and_resize_image(
                image_path=str(image_path),
                images_across=2,
                images_high=2,
                output_size=50,
                custom_folder="cuts_margin",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_161616",
                output_base_dir=tmp_path / "out",
                crop_margin=5,
            )

            rows = _read_tracking_rows(out / "sheet_tracking_index.csv")
            self.assertEqual(rows, [
                {"code": "A1", "filename": "A1.png", "left": "-5", "top": "-5", "right": "45", "bottom": "35", "width": "50", "height": "40"},
                {"code": "A2", "filename": "A2.png", "left": "35", "top": "-5", "right": "85", "bottom": "35", "width": "50", "height": "40"},
                {"code": "B1", "filename": "B1.png", "left": "-5", "top": "25", "right": "45", "bottom": "65", "width": "50", "height": "40"},
                {"code": "B2", "filename": "B2.png", "left": "35", "top": "25", "right": "85", "bottom": "65", "width": "50", "height": "40"},
            ])

            with Image.open(out / "A1.png") as crop:
                self.assertEqual(crop.size, (50, 40))
                self.assertEqual(crop.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(crop.getpixel((25, 20)), (0, 0, 0))

            diagnostics = _read_diagnostics(out / "sheet_diagnostics.json")
            self.assertEqual(diagnostics["crop_margin_px"], 5)
            self.assertIn("auto_straightened_crops", diagnostics)
            self.assertIn("max_abs_straightening_deg", diagnostics)
            self.assertIn("remaining_overlap_pairs", diagnostics)

    def test_split_and_resize_quality_max_keeps_native_crop_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"
            _save_solid_image(image_path, size=(80, 60), image_format="PNG")

            out = core.split_and_resize_image(
                image_path=str(image_path),
                images_across=2,
                images_high=2,
                output_size=None,
                custom_folder="cuts_native",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_171717",
                output_base_dir=tmp_path / "out",
                crop_margin=0,
            )

            with Image.open(out / "A1.png") as crop:
                self.assertEqual(crop.size, (40, 30))

    def test_split_and_resize_margin_zero_trims_white_border_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"

            img = Image.new("RGB", (90, 70), "white")
            draw = ImageDraw.Draw(img)
            draw.rectangle((10, 8, 79, 61), fill=(0, 0, 0))
            img.save(image_path, format="PNG")

            out = core.split_and_resize_image(
                image_path=str(image_path),
                images_across=1,
                images_high=1,
                output_size=None,
                custom_folder="cuts_trim",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_181818",
                output_base_dir=tmp_path / "out",
                crop_margin=0,
            )

            with Image.open(out / "A1.png") as crop:
                self.assertEqual(crop.size, (70, 54))
                self.assertLess(max(crop.getpixel((0, 0))), 30)
                self.assertLess(max(crop.getpixel((69, 53))), 30)

            rows = _read_tracking_rows(out / "sheet_tracking_index.csv")
            self.assertEqual(rows[0]["left"], "10")
            self.assertEqual(rows[0]["top"], "8")
            self.assertEqual(rows[0]["right"], "80")
            self.assertEqual(rows[0]["bottom"], "62")

            diagnostics = _read_diagnostics(out / "sheet_diagnostics.json")
            self.assertEqual(diagnostics["white_border_trimmed_crops"], 1)
            self.assertEqual(diagnostics["white_border_residual_crops"], 0)
            self.assertGreaterEqual(diagnostics["max_white_border_trim_px"], 8)

    def test_split_and_resize_high_margin_enables_overlap_guard_and_removes_intersections(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "sheet.png"
            _save_solid_image(image_path, size=(100, 40), image_format="PNG")

            out = core.split_and_resize_image(
                image_path=str(image_path),
                images_across=2,
                images_high=1,
                output_size=None,
                custom_folder="cuts_overlap",
                maintain_format=True,
                smart_grid=False,
                timestamp="20260302_191919",
                output_base_dir=tmp_path / "out",
                crop_margin=30,
            )

            rows = _read_tracking_rows(out / "sheet_tracking_index.csv")
            right_a1 = int(rows[0]["right"])
            left_a2 = int(rows[1]["left"])
            self.assertLessEqual(right_a1, left_a2)

            diagnostics = _read_diagnostics(out / "sheet_diagnostics.json")
            self.assertTrue(diagnostics["overlap_guard_active"])
            self.assertGreater(diagnostics["overlap_corrected_pairs"], 0)
            self.assertGreater(diagnostics["max_overlap_before_correction_px"], 0)
            self.assertEqual(diagnostics["remaining_overlap_pairs"], 0)

    def test_assess_image_croppability_rejects_invalid_grid(self):
        is_ok, reason, diagnostics = core.assess_image_croppability(
            image_path="does_not_matter.jpg",
            images_across=0,
            images_high=2,
            smart_grid=True,
        )
        self.assertFalse(is_ok)
        self.assertEqual(reason, "invalid_grid")
        self.assertIsNone(diagnostics)

    def test_assess_image_croppability_short_circuit_for_single_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "single.png"
            _save_solid_image(image_path, size=(80, 60), image_format="PNG")

            is_ok, reason, diagnostics = core.assess_image_croppability(
                image_path=str(image_path),
                images_across=1,
                images_high=1,
                smart_grid=True,
            )
            self.assertTrue(is_ok)
            self.assertEqual(reason, "classic_or_single_cell")
            self.assertIsNone(diagnostics)

    def test_assess_image_croppability_reports_read_errors(self):
        is_ok, reason, diagnostics = core.assess_image_croppability(
            image_path="c:/this/path/does/not/exist.png",
            images_across=2,
            images_high=2,
            smart_grid=True,
        )
        self.assertFalse(is_ok)
        self.assertTrue(reason.startswith("read_error:"))
        self.assertIsNone(diagnostics)

    def test_assess_and_archive_impossible_files_archives_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            impossible_path = tmp_path / "white.jpg"
            processable_path = tmp_path / "ok.png"
            Image.new("RGB", (800, 800), "white").save(impossible_path, format="JPEG")
            _save_solid_image(processable_path, size=(80, 60), image_format="PNG")

            jobs = [
                {
                    "image_path": str(impossible_path),
                    "images_across": 4,
                    "images_high": 4,
                    "smart_grid": True,
                    "marker": "bad",
                },
                {
                    "image_path": str(processable_path),
                    "images_across": 1,
                    "images_high": 1,
                    "smart_grid": True,
                    "marker": "ok",
                },
            ]

            processable, impossible, report_path = core.assess_and_archive_impossible_files(
                image_jobs=jobs,
                archive_folder=tmp_path / "archive",
                min_confidence=12.0,
            )

            self.assertEqual(len(processable), 1)
            self.assertEqual(processable[0]["marker"], "ok")
            self.assertEqual(len(impossible), 1)
            self.assertEqual(impossible[0]["marker"], "bad")
            self.assertTrue(Path(impossible[0]["archived_path"]).exists())
            self.assertTrue(report_path.exists())

            with report_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["reason"])

    def test_assess_and_archive_impossible_files_renames_duplicate_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src_a = tmp_path / "a"
            src_b = tmp_path / "b"
            src_a.mkdir()
            src_b.mkdir()

            path_a = src_a / "duplicate.jpg"
            path_b = src_b / "duplicate.jpg"
            Image.new("RGB", (800, 800), "white").save(path_a, format="JPEG")
            Image.new("RGB", (800, 800), "white").save(path_b, format="JPEG")

            jobs = [
                {"image_path": str(path_a), "images_across": 4, "images_high": 4, "smart_grid": True},
                {"image_path": str(path_b), "images_across": 4, "images_high": 4, "smart_grid": True},
            ]

            processable, impossible, report_path = core.assess_and_archive_impossible_files(
                image_jobs=jobs,
                archive_folder=tmp_path / "archive",
                min_confidence=12.0,
            )

            self.assertEqual(processable, [])
            self.assertEqual(len(impossible), 2)
            archived_names = sorted(Path(job["archived_path"]).name for job in impossible)
            self.assertEqual(archived_names, ["duplicate.jpg", "duplicate_2.jpg"])
            self.assertTrue(report_path.exists())

    def test_auto_straighten_crop_reduces_measured_skew(self):
        skewed = _make_rotated_content_image(rotation_deg=8.0)
        before = abs(core._estimate_crop_skew_degrees(skewed))
        corrected, estimated, applied = core._auto_straighten_crop(skewed, min_angle=0.2, max_angle=15.0)
        after = abs(core._estimate_crop_skew_degrees(corrected))

        self.assertTrue(applied)
        self.assertGreater(abs(estimated), 0.2)
        self.assertLess(after, before)

    @unittest.skipUnless(core.CV_AVAILABLE, "OpenCV not available")
    def test_detect_smart_grid_boxes_supports_irregular_block_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "irregular.jpg"
            _make_irregular_grid_image(
                image_path,
                col_widths=[120, 180, 140, 210],
                row_heights=[100, 170, 130],
                gap=14,
                margin=22,
                image_format="JPEG",
            )

            with Image.open(image_path) as img:
                boxes, diagnostics = core.detect_smart_grid_boxes(
                    img,
                    images_across=4,
                    images_high=3,
                    return_diagnostics=True,
                )

            self.assertIsNotNone(boxes)
            self.assertEqual(len(boxes), 12)
            self.assertEqual((diagnostics or {}).get("method"), "direct_detected")

            widths = sorted({right - left for left, top, right, bottom in boxes})
            heights = sorted({bottom - top for left, top, right, bottom in boxes})
            self.assertGreater(len(widths), 1)
            self.assertGreater(len(heights), 1)

    @unittest.skipUnless(core.CV_AVAILABLE, "OpenCV not available")
    def test_assess_image_croppability_accepts_irregular_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "irregular.jpg"
            _make_irregular_grid_image(
                image_path,
                col_widths=[110, 190, 135, 200],
                row_heights=[90, 165, 145],
                gap=12,
                margin=20,
                image_format="JPEG",
            )

            is_ok, reason, diagnostics = core.assess_image_croppability(
                image_path=str(image_path),
                images_across=4,
                images_high=3,
                smart_grid=True,
                min_confidence=12.0,
            )
            self.assertTrue(is_ok)
            self.assertEqual(reason, "ok")
            self.assertIsInstance(diagnostics, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
