import csv
import json
import shutil
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

Rect = Tuple[int, int, int, int]

try:
    import cv2
    import numpy as np

    CV_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV_AVAILABLE = False


def is_cv_available():
    return CV_AVAILABLE


def resize_image_keep_aspect_ratio(image, target_size):
    """Resize image while preserving the original aspect ratio."""
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = target_size
        new_height = int(target_size * original_height / original_width)
    else:
        new_height = target_size
        new_width = int(target_size * original_width / original_height)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def generate_uniform_grid_boxes(img_width, img_height, images_across, images_high):
    """Generate evenly spaced crop boxes in row-major order."""
    boxes = []
    for row in range(images_high):
        top = int(round(row * img_height / images_high))
        bottom = int(round((row + 1) * img_height / images_high))
        for col in range(images_across):
            left = int(round(col * img_width / images_across))
            right = int(round((col + 1) * img_width / images_across))
            boxes.append((left, top, right, bottom))
    return boxes


def _smooth_profile(values, kernel_size):
    if not CV_AVAILABLE:
        return values
    kernel_size = int(max(1, kernel_size))
    if kernel_size <= 1:
        return values.astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(values, kernel, mode="same")


def _find_high_bands(profile, threshold, min_width):
    mask = profile >= threshold
    bands = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        end = idx - 1
        if end - start + 1 >= min_width:
            bands.append((start, end))
    return bands


def _merge_nearby_bands(bands, max_gap):
    if not bands:
        return []
    merged = [list(bands[0])]
    for start, end in bands[1:]:
        if start - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(entry) for entry in merged]


def _ensure_edge_bands(bands, axis_len):
    if not bands:
        return [(0, 0), (axis_len - 1, axis_len - 1)]

    result = list(bands)
    edge_zone = max(6, int(axis_len * 0.07))
    first_center = (result[0][0] + result[0][1]) / 2.0
    last_center = (result[-1][0] + result[-1][1]) / 2.0

    if first_center > edge_zone:
        result.insert(0, (0, 0))
    if last_center < (axis_len - 1 - edge_zone):
        result.append((axis_len - 1, axis_len - 1))
    return result


def _select_expected_bands(bands, profile, expected_count, axis_len):
    if expected_count <= 0:
        return []

    bands = sorted(bands, key=lambda band: band[0])
    bands = _ensure_edge_bands(bands, axis_len)

    if len(bands) < expected_count:
        return None
    if len(bands) == expected_count:
        return bands

    # Prefer strongest separators while preserving edge separators.
    # This avoids forcing regular spacing, so non-uniform grids remain valid.
    strengths = {
        idx: float(np.mean(profile[start : end + 1]))
        for idx, (start, end) in enumerate(bands)
    }
    selected = list(range(len(bands)))
    locked = set()
    if len(selected) >= 1:
        locked.add(0)
    if len(selected) >= 2:
        locked.add(len(selected) - 1)

    while len(selected) > expected_count:
        removable = [idx for idx in selected if idx not in locked]
        if not removable:
            removable = list(selected)
        weakest = min(removable, key=lambda idx: strengths.get(idx, 0.0))
        selected.remove(weakest)
        locked.discard(weakest)

    return [bands[idx] for idx in sorted(selected)]


def _detect_separator_observations(gray, splits, axis):
    if not CV_AVAILABLE:
        return None, None, None

    axis_len = gray.shape[1] if axis == "x" else gray.shape[0]
    white_mask = gray > 215
    profile = white_mask.mean(axis=0 if axis == "x" else 1).astype(np.float32)
    profile = _smooth_profile(profile, max(5, axis_len // 200))

    threshold = max(0.55, float(np.quantile(profile, 0.80)))
    min_width = max(3, axis_len // 200)
    bands = _find_high_bands(profile, threshold, min_width)
    bands = _merge_nearby_bands(bands, max_gap=max(2, axis_len // 80))
    return bands, profile, axis_len


def _detect_separator_bands(gray, splits, axis):
    bands, profile, axis_len = _detect_separator_observations(gray, splits, axis)
    if bands is None:
        return None
    expected_count = splits + 1

    return _select_expected_bands(bands, profile, expected_count, axis_len)


def _band_centers_and_widths(bands):
    centers = np.array([(start + end) / 2.0 for start, end in bands], dtype=np.float32)
    widths = np.array([max(1.0, float(end - start + 1)) for start, end in bands], dtype=np.float32)
    return centers, widths


def _fit_regular_bands(observed_bands, expected_count, axis_len):
    """
    Fit a regular separator sequence from incomplete/noisy observed bands.

    This allows inferring missing lines when parts of the sheet are painted or out of frame.
    """
    if not CV_AVAILABLE or not observed_bands or expected_count < 2:
        return None

    centers, widths = _band_centers_and_widths(observed_bands)
    if len(centers) == 0:
        return None

    tolerance = max(3.0, axis_len * 0.015)
    index_pairs = [(i, j) for i in range(expected_count) for j in range(i + 1, expected_count)]

    best_expected = None
    best_score = None

    if len(centers) >= 2:
        for c_idx in range(len(centers)):
            for c_jdx in range(c_idx + 1, len(centers)):
                c1 = float(centers[c_idx])
                c2 = float(centers[c_jdx])
                if abs(c2 - c1) < 1.0:
                    continue
                for i, j in index_pairs:
                    step = (c2 - c1) / float(j - i)
                    if step <= 2.0:
                        continue
                    intercept = c1 - (i * step)
                    expected = intercept + (step * np.arange(expected_count, dtype=np.float32))

                    residuals = np.min(np.abs(centers[:, None] - expected[None, :]), axis=1)
                    inliers = int(np.sum(residuals <= tolerance))
                    mean_res = float(np.mean(residuals))
                    out_of_frame = float(np.sum(np.maximum(0.0, -expected) + np.maximum(0.0, expected - (axis_len - 1))))

                    score = (inliers, -mean_res, -out_of_frame)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_expected = expected

    if best_expected is None:
        # Last-resort regular model.
        best_expected = np.linspace(0, axis_len - 1, expected_count, dtype=np.float32)

    # Refine model by snapping observed centers to nearest separator index and fitting a line.
    nearest_idx = np.argmin(np.abs(centers[:, None] - best_expected[None, :]), axis=1)
    nearest_res = np.min(np.abs(centers[:, None] - best_expected[None, :]), axis=1)
    valid = nearest_res <= (tolerance * 1.5)
    if np.any(valid):
        xs = nearest_idx[valid].astype(np.float32)
        ys = centers[valid].astype(np.float32)
        if len(np.unique(xs)) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)
            if slope > 2.0:
                best_expected = intercept + (slope * np.arange(expected_count, dtype=np.float32))

    median_width = float(np.median(widths)) if len(widths) else max(4.0, axis_len * 0.015)
    median_width = max(4.0, min(median_width, axis_len / max(2.0, expected_count)))

    half = median_width / 2.0
    fitted_bands = []
    for center in best_expected:
        start = int(round(center - half))
        end = int(round(center + half))
        if end <= start:
            end = start + 1
        fitted_bands.append((start, end))

    return fitted_bands


def _intervals_from_bands(bands):
    intervals = []
    for idx in range(len(bands) - 1):
        start = bands[idx][1] + 1
        end = bands[idx + 1][0]
        if end - start > 4:
            intervals.append((start, end))
    return intervals


def _trim_cell_bounds(gray, left, top, right, bottom):
    """Trim residual white border while avoiding over-trim on painted edges."""
    img_h, img_w = gray.shape[:2]
    clip_left = max(0, left)
    clip_top = max(0, top)
    clip_right = min(img_w, right)
    clip_bottom = min(img_h, bottom)

    cell = gray[clip_top:clip_bottom, clip_left:clip_right]
    if cell.size == 0:
        return left, top, right, bottom

    non_white_mask = cell < 235
    col_ratio = non_white_mask.mean(axis=0)
    row_ratio = non_white_mask.mean(axis=1)

    active_cols = np.where(col_ratio > 0.02)[0]
    active_rows = np.where(row_ratio > 0.02)[0]
    if len(active_cols) == 0 or len(active_rows) == 0:
        return left, top, right, bottom

    local_left = int(active_cols[0])
    local_right = int(active_cols[-1]) + 1
    local_top = int(active_rows[0])
    local_bottom = int(active_rows[-1]) + 1

    cell_width = clip_right - clip_left
    cell_height = clip_bottom - clip_top
    max_trim_x = max(1, int(cell_width * 0.22))
    max_trim_y = max(1, int(cell_height * 0.22))

    local_left = min(local_left, max_trim_x)
    local_top = min(local_top, max_trim_y)
    local_right = max(local_right, cell_width - max_trim_x)
    local_bottom = max(local_bottom, cell_height - max_trim_y)

    pad = 1
    trimmed_left = clip_left + max(0, local_left - pad)
    trimmed_top = clip_top + max(0, local_top - pad)
    trimmed_right = clip_left + min(cell_width, local_right + pad)
    trimmed_bottom = clip_top + min(cell_height, local_bottom + pad)

    if trimmed_right - trimmed_left < 8 or trimmed_bottom - trimmed_top < 8:
        return left, top, right, bottom
    return trimmed_left, trimmed_top, trimmed_right, trimmed_bottom


def _compute_grid_diagnostics(crop_boxes, image_size, method, observed_x_count, observed_y_count, expected_x, expected_y):
    img_w, img_h = image_size
    widths = np.array([max(1, right - left) for left, top, right, bottom in crop_boxes], dtype=np.float32)
    heights = np.array([max(1, bottom - top) for left, top, right, bottom in crop_boxes], dtype=np.float32)

    width_cv = float(np.std(widths) / max(1.0, np.mean(widths)))
    height_cv = float(np.std(heights) / max(1.0, np.mean(heights)))
    malformed_cells = int(np.sum(np.abs(widths - np.median(widths)) > (0.25 * max(1.0, np.median(widths)))))
    malformed_cells += int(np.sum(np.abs(heights - np.median(heights)) > (0.25 * max(1.0, np.median(heights)))))

    offscreen_boxes = 0
    for left, top, right, bottom in crop_boxes:
        if left < 0 or top < 0 or right > img_w or bottom > img_h:
            offscreen_boxes += 1

    inferred_x = max(0, expected_x - observed_x_count)
    inferred_y = max(0, expected_y - observed_y_count)

    # Heuristic confidence score (0..100).
    score = 100.0
    score -= min(25.0, 120.0 * width_cv)
    score -= min(25.0, 120.0 * height_cv)
    score -= min(30.0, offscreen_boxes * 4.0)
    score -= min(20.0, (inferred_x + inferred_y) * 2.0)
    score = max(0.0, min(100.0, score))

    return {
        "method": method,
        "expected_vertical_separators": int(expected_x),
        "expected_horizontal_separators": int(expected_y),
        "observed_vertical_separators": int(observed_x_count),
        "observed_horizontal_separators": int(observed_y_count),
        "inferred_vertical_separators": int(inferred_x),
        "inferred_horizontal_separators": int(inferred_y),
        "offscreen_cells": int(offscreen_boxes),
        "cell_width_cv": round(width_cv, 6),
        "cell_height_cv": round(height_cv, 6),
        "malformed_cell_flags": int(malformed_cells),
        "confidence_score": round(score, 2),
    }


def detect_smart_grid_boxes(image, images_across, images_high, return_diagnostics=False):
    """
    Detect real cell bounds from scan-like grids with imperfect white borders.

    Returns a list of crop boxes in row-major order, or None on detection failure.
    """
    if not CV_AVAILABLE:
        return None

    rgb_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    observed_vertical_bands, vertical_profile, axis_w = _detect_separator_observations(gray, images_across, axis="x")
    observed_horizontal_bands, horizontal_profile, axis_h = _detect_separator_observations(gray, images_high, axis="y")

    if observed_vertical_bands is None or observed_horizontal_bands is None:
        return (None, None) if return_diagnostics else None

    expected_v = images_across + 1
    expected_h = images_high + 1

    selected_vertical = _select_expected_bands(observed_vertical_bands, vertical_profile, expected_v, axis_w)
    selected_horizontal = _select_expected_bands(observed_horizontal_bands, horizontal_profile, expected_h, axis_h)
    fitted_vertical = _fit_regular_bands(observed_vertical_bands, expected_v, axis_w)
    fitted_horizontal = _fit_regular_bands(observed_horizontal_bands, expected_h, axis_h)

    method = "direct_detected"
    use_regular_model = False

    final_vertical = selected_vertical
    final_horizontal = selected_horizontal

    if not final_vertical or len(final_vertical) != expected_v:
        final_vertical = fitted_vertical
        use_regular_model = True
    if not final_horizontal or len(final_horizontal) != expected_h:
        final_horizontal = fitted_horizontal
        use_regular_model = True

    if not final_vertical or not final_horizontal:
        return (None, None) if return_diagnostics else None

    x_intervals = _intervals_from_bands(final_vertical)
    y_intervals = _intervals_from_bands(final_horizontal)

    if len(x_intervals) != images_across or len(y_intervals) != images_high:
        if fitted_vertical and fitted_horizontal:
            method = "fitted_regular_grid"
            use_regular_model = True
            x_intervals = _intervals_from_bands(fitted_vertical)
            y_intervals = _intervals_from_bands(fitted_horizontal)

    if len(x_intervals) != images_across or len(y_intervals) != images_high:
        return (None, None) if return_diagnostics else None

    boxes = []
    for top, bottom in y_intervals:
        for left, right in x_intervals:
            if use_regular_model:
                boxes.append((left, top, right, bottom))
            else:
                boxes.append(_trim_cell_bounds(gray, left, top, right, bottom))

    expected_parts = images_across * images_high
    if len(boxes) != expected_parts:
        return (None, None) if return_diagnostics else None

    if use_regular_model:
        method = "fitted_regular_grid"

    diagnostics = _compute_grid_diagnostics(
        boxes,
        image_size=image.size,
        method=method,
        observed_x_count=len(observed_vertical_bands),
        observed_y_count=len(observed_horizontal_bands),
        expected_x=expected_v,
        expected_y=expected_h,
    )

    return (boxes, diagnostics) if return_diagnostics else boxes


def create_output_folder(image_path, custom_folder, images_across, images_high, timestamp, output_base_dir=None):
    """Create output folder path in a cross-platform compatible way."""
    image_path = Path(image_path)
    base_name = image_path.stem
    source_directory = Path(output_base_dir) if output_base_dir else image_path.parent

    if images_across == 1 and images_high == 1:
        if custom_folder:
            output_folder = source_directory / custom_folder / timestamp
        else:
            output_folder = source_directory / timestamp
    else:
        if custom_folder:
            output_folder = source_directory / custom_folder / base_name
        else:
            output_folder = source_directory / base_name

    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def _resolve_output_format(image, maintain_format):
    if maintain_format and image.format:
        image_format = image.format.upper()
    else:
        image_format = "JPEG"

    if image_format == "JPG":
        image_format = "JPEG"
    if image_format == "MPO":
        image_format = "JPEG"

    extension_map = {
        "JPEG": "jpg",
        "PNG": "png",
        "WEBP": "webp",
        "BMP": "bmp",
        "TIFF": "tif",
    }
    extension = extension_map.get(image_format, image_format.lower())
    return image_format, extension


def _index_to_alpha(index):
    letters = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters = chr(65 + remainder) + letters
        if value == 0:
            break
        value -= 1
    return letters


def _generate_crop_codes(images_across, images_high, total_parts):
    expected_parts = images_across * images_high
    if total_parts != expected_parts:
        return [f"P{idx + 1}" for idx in range(total_parts)]

    codes = []
    for row in range(images_high):
        row_label = _index_to_alpha(row)
        for col in range(images_across):
            codes.append(f"{row_label}{col + 1}")
    return codes


def _load_overlay_font(image_height):
    font_size = max(14, int(image_height * 0.025))
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_tracking_overlay(source_image, output_folder, source_stem, codes, crop_boxes):
    overlay = source_image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = _load_overlay_font(overlay.height)
    stroke_width = max(2, int(min(overlay.size) * 0.003))
    text_pad = max(4, stroke_width + 1)

    for code, (left, top, right, bottom) in zip(codes, crop_boxes):
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=stroke_width)

        text_bbox = draw.textbbox((0, 0), code, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_left = min(max(0, left + stroke_width), max(0, overlay.width - text_width - (2 * text_pad)))
        label_top = min(max(0, top + stroke_width), max(0, overlay.height - text_height - (2 * text_pad)))
        label_right = label_left + text_width + (2 * text_pad)
        label_bottom = label_top + text_height + (2 * text_pad)

        draw.rectangle((label_left, label_top, label_right, label_bottom), fill=(255, 255, 255), outline=(255, 0, 0))
        draw.text((label_left + text_pad, label_top + text_pad), code, fill=(0, 0, 0), font=font)

    overlay_path = output_folder / f"{source_stem}_tracking_overlay.jpg"
    overlay.save(overlay_path, "JPEG", quality=95)
    return overlay_path


def _save_tracking_index(output_folder, source_stem, codes, crop_boxes, extension):
    index_path = output_folder / f"{source_stem}_tracking_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "filename", "left", "top", "right", "bottom", "width", "height"])
        for code, (left, top, right, bottom) in zip(codes, crop_boxes):
            width = right - left
            height = bottom - top
            writer.writerow([code, f"{code}.{extension}", left, top, right, bottom, width, height])
    return index_path


def _save_diagnostics(output_folder, source_stem, diagnostics, total_parts, smart_grid_requested):
    diagnostics_path = output_folder / f"{source_stem}_diagnostics.json"
    payload = dict(diagnostics or {})
    payload["total_parts"] = int(total_parts)
    payload["smart_grid_requested"] = bool(smart_grid_requested)
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return diagnostics_path


def _white_fill_for_mode(mode):
    if mode == "L":
        return 255
    if mode == "LA":
        return (255, 255)
    if mode == "RGBA":
        return (255, 255, 255, 255)
    if mode == "CMYK":
        return (0, 0, 0, 0)
    return (255, 255, 255)


def _crop_with_white_padding(image, box):
    """Crop image with white padding when the box exceeds image boundaries."""
    left, top, right, bottom = [int(v) for v in box]
    width = max(1, right - left)
    height = max(1, bottom - top)

    canvas = Image.new(image.mode, (width, height), _white_fill_for_mode(image.mode))
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.width, right)
    src_bottom = min(image.height, bottom)

    if src_right > src_left and src_bottom > src_top:
        patch = image.crop((src_left, src_top, src_right, src_bottom))
        paste_x = src_left - left
        paste_y = src_top - top
        canvas.paste(patch, (paste_x, paste_y))

    return canvas


def _expand_box_with_margin(box, margin):
    """Expand a crop box by a symmetric pixel margin."""
    margin = int(max(0, margin))
    if margin == 0:
        return tuple(int(v) for v in box)

    left, top, right, bottom = [int(v) for v in box]
    return (left - margin, top - margin, right + margin, bottom + margin)


def _edge_white_ratios(image, white_threshold=240):
    gray_image = image.convert("L")
    width, height = gray_image.size
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0

    if np is not None:
        gray = np.array(gray_image, dtype=np.uint8)
        top_ratio = float(np.mean(gray[0, :] >= white_threshold))
        bottom_ratio = float(np.mean(gray[-1, :] >= white_threshold))
        left_ratio = float(np.mean(gray[:, 0] >= white_threshold))
        right_ratio = float(np.mean(gray[:, -1] >= white_threshold))
        return top_ratio, bottom_ratio, left_ratio, right_ratio

    pixels = gray_image.load()

    top_white = 0
    bottom_white = 0
    for x in range(width):
        if int(pixels[x, 0]) >= white_threshold:
            top_white += 1
        if int(pixels[x, height - 1]) >= white_threshold:
            bottom_white += 1

    left_white = 0
    right_white = 0
    for y in range(height):
        if int(pixels[0, y]) >= white_threshold:
            left_white += 1
        if int(pixels[width - 1, y]) >= white_threshold:
            right_white += 1

    return (
        top_white / float(width),
        bottom_white / float(width),
        left_white / float(height),
        right_white / float(height),
    )


def _has_strong_white_border(image, white_threshold=240, min_white_ratio=0.97):
    top_ratio, bottom_ratio, left_ratio, right_ratio = _edge_white_ratios(image, white_threshold=white_threshold)
    return any(ratio >= float(min_white_ratio) for ratio in (top_ratio, bottom_ratio, left_ratio, right_ratio))


def _trim_white_borders_from_crop(
    image,
    max_trim_ratio=0.22,
    white_threshold=240,
    min_white_ratio=0.97,
):
    """
    Trim strong white borders from a crop without over-cutting the useful content.

    Returns:
        (refined_image, (left_trim, top_trim, right_trim, bottom_trim), trimmed_bool)
    """
    gray_image = image.convert("L")
    width, height = gray_image.size
    if width < 8 or height < 8:
        return image, (0, 0, 0, 0), False

    if np is not None:
        gray = np.array(gray_image, dtype=np.uint8)
        if gray.size == 0:
            return image, (0, 0, 0, 0), False

        non_white_ratio = float(np.mean(gray < max(0, white_threshold - 14)))
        if non_white_ratio < 0.01:
            return image, (0, 0, 0, 0), False

        white_mask = gray >= white_threshold
        col_white_ratio = white_mask.mean(axis=0)
        row_white_ratio = white_mask.mean(axis=1)
    else:
        pixels = gray_image.load()
        non_white_pixels = 0
        col_white_counts = [0] * width
        row_white_counts = [0] * height
        for y in range(height):
            for x in range(width):
                val = int(pixels[x, y])
                if val < max(0, white_threshold - 14):
                    non_white_pixels += 1
                if val >= white_threshold:
                    col_white_counts[x] += 1
                    row_white_counts[y] += 1

        non_white_ratio = non_white_pixels / float(width * height)
        if non_white_ratio < 0.01:
            return image, (0, 0, 0, 0), False

        col_white_ratio = [count / float(height) for count in col_white_counts]
        row_white_ratio = [count / float(width) for count in row_white_counts]

    max_trim_x = max(1, int(width * float(max_trim_ratio)))
    max_trim_y = max(1, int(height * float(max_trim_ratio)))

    left_trim = 0
    while left_trim < max_trim_x and left_trim < (width - 2) and float(col_white_ratio[left_trim]) >= float(min_white_ratio):
        left_trim += 1

    right_trim = 0
    while right_trim < max_trim_x and right_trim < (width - 2):
        idx = width - 1 - right_trim
        if float(col_white_ratio[idx]) < float(min_white_ratio):
            break
        right_trim += 1

    top_trim = 0
    while top_trim < max_trim_y and top_trim < (height - 2) and float(row_white_ratio[top_trim]) >= float(min_white_ratio):
        top_trim += 1

    bottom_trim = 0
    while bottom_trim < max_trim_y and bottom_trim < (height - 2):
        idx = height - 1 - bottom_trim
        if float(row_white_ratio[idx]) < float(min_white_ratio):
            break
        bottom_trim += 1

    if left_trim == 0 and right_trim == 0 and top_trim == 0 and bottom_trim == 0:
        return image, (0, 0, 0, 0), False

    remaining_width = width - left_trim - right_trim
    remaining_height = height - top_trim - bottom_trim
    min_keep_width = max(8, int(width * 0.35))
    min_keep_height = max(8, int(height * 0.35))
    if remaining_width < min_keep_width or remaining_height < min_keep_height:
        return image, (0, 0, 0, 0), False

    refined = image.crop((left_trim, top_trim, width - right_trim, height - bottom_trim))
    return refined, (left_trim, top_trim, right_trim, bottom_trim), True


def _median_int(values):
    if not values:
        return 0
    ordered = sorted(int(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2.0))


def _should_enable_overlap_guard(crop_margin, crop_boxes):
    crop_margin = int(max(0, crop_margin or 0))
    if crop_margin <= 0 or len(crop_boxes) <= 1:
        return False

    spans = []
    for left, top, right, bottom in crop_boxes:
        spans.append(max(1, int(right - left)))
        spans.append(max(1, int(bottom - top)))

    typical_span = max(1, _median_int(spans))
    trigger_margin = max(8, int(round(typical_span * 0.09)))
    return crop_margin >= trigger_margin


def _separate_horizontal_pair(left_box, right_box, min_side=8):
    overlap = int(left_box[2] - right_box[0])
    if overlap <= 0:
        return False

    min_split = int(left_box[0] + min_side)
    max_split = int(right_box[2] - min_side)
    if min_split > max_split:
        min_split = int(left_box[0] + 1)
        max_split = int(right_box[2] - 1)
        if min_split > max_split:
            return False

    split = int(round((left_box[2] + right_box[0]) / 2.0))
    split = max(min_split, min(max_split, split))
    left_box[2] = split
    right_box[0] = split
    return True


def _separate_vertical_pair(top_box, bottom_box, min_side=8):
    overlap = int(top_box[3] - bottom_box[1])
    if overlap <= 0:
        return False

    min_split = int(top_box[1] + min_side)
    max_split = int(bottom_box[3] - min_side)
    if min_split > max_split:
        min_split = int(top_box[1] + 1)
        max_split = int(bottom_box[3] - 1)
        if min_split > max_split:
            return False

    split = int(round((top_box[3] + bottom_box[1]) / 2.0))
    split = max(min_split, min(max_split, split))
    top_box[3] = split
    bottom_box[1] = split
    return True


def _count_adjacent_overlaps(crop_boxes, images_across, images_high):
    overlaps = 0
    max_overlap = 0

    for row in range(images_high):
        base = row * images_across
        for col in range(images_across - 1):
            left_box = crop_boxes[base + col]
            right_box = crop_boxes[base + col + 1]
            overlap = int(left_box[2] - right_box[0])
            if overlap > 0:
                overlaps += 1
                max_overlap = max(max_overlap, overlap)

    for row in range(images_high - 1):
        for col in range(images_across):
            top_box = crop_boxes[(row * images_across) + col]
            bottom_box = crop_boxes[((row + 1) * images_across) + col]
            overlap = int(top_box[3] - bottom_box[1])
            if overlap > 0:
                overlaps += 1
                max_overlap = max(max_overlap, overlap)

    return overlaps, max_overlap


def _resolve_adjacent_box_overlaps(crop_boxes, images_across, images_high, min_side=8, max_passes=4):
    """
    Enforce non-overlap across adjacent cells when large margins create intersections.

    Returns:
        (resolved_boxes, corrected_pairs, max_overlap_before_px, remaining_overlap_pairs, remaining_max_overlap_px)
    """
    if len(crop_boxes) <= 1:
        return list(crop_boxes), 0, 0, 0, 0

    boxes = [[int(v) for v in box] for box in crop_boxes]
    corrected_pairs = 0
    max_overlap_before = 0

    for _ in range(max(1, int(max_passes))):
        changed = False

        for row in range(images_high):
            base = row * images_across
            for col in range(images_across - 1):
                left_box = boxes[base + col]
                right_box = boxes[base + col + 1]
                overlap = int(left_box[2] - right_box[0])
                if overlap <= 0:
                    continue
                max_overlap_before = max(max_overlap_before, overlap)
                if _separate_horizontal_pair(left_box, right_box, min_side=min_side):
                    corrected_pairs += 1
                    changed = True

        for row in range(images_high - 1):
            for col in range(images_across):
                top_box = boxes[(row * images_across) + col]
                bottom_box = boxes[((row + 1) * images_across) + col]
                overlap = int(top_box[3] - bottom_box[1])
                if overlap <= 0:
                    continue
                max_overlap_before = max(max_overlap_before, overlap)
                if _separate_vertical_pair(top_box, bottom_box, min_side=min_side):
                    corrected_pairs += 1
                    changed = True

        if not changed:
            break

    resolved = [tuple(box) for box in boxes]
    remaining_pairs, remaining_max = _count_adjacent_overlaps(resolved, images_across, images_high)
    return resolved, corrected_pairs, max_overlap_before, remaining_pairs, remaining_max


def _estimate_crop_skew_degrees(image):
    """
    Estimate crop skew angle in degrees.

    Returns signed angle where positive means clockwise tilt of the content.
    """
    if not CV_AVAILABLE:
        return 0.0

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    # Focus on non-white content to avoid background noise from scan paper.
    mask = gray < 235
    if int(np.count_nonzero(mask)) < max(25, int(mask.size * 0.01)):
        return 0.0

    ys, xs = np.where(mask)
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    if len(points) < 10:
        return 0.0

    rect = cv2.minAreaRect(points)
    (_, _), (width, height), angle = rect
    if width <= 1 or height <= 1:
        return 0.0

    # minAreaRect returns angle in [-90, 0). Convert to near-horizontal deviation.
    if width < height:
        angle = angle + 90.0
    while angle <= -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    return float(angle)


def _rotate_image_keep_size(image, degrees):
    """Rotate image around center and keep original size, padding with white."""
    if abs(float(degrees)) < 1e-6:
        return image

    original_size = image.size
    rotated = image.rotate(degrees, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=_white_fill_for_mode(image.mode))

    canvas = Image.new(image.mode, original_size, _white_fill_for_mode(image.mode))
    paste_x = (original_size[0] - rotated.width) // 2
    paste_y = (original_size[1] - rotated.height) // 2
    canvas.paste(rotated, (paste_x, paste_y))
    return canvas


def _auto_straighten_crop(image, min_angle=0.35, max_angle=10.0):
    """
    Autonomously straighten slightly tilted crops.

    Returns:
        (corrected_image, estimated_angle_deg, applied_bool)
    """
    angle = float(_estimate_crop_skew_degrees(image))
    abs_angle = abs(angle)
    if abs_angle < float(min_angle) or abs_angle > float(max_angle):
        return image, angle, False

    original_residual = abs_angle
    corrected_a = _rotate_image_keep_size(image, -angle)
    corrected_b = _rotate_image_keep_size(image, angle)
    residual_a = abs(float(_estimate_crop_skew_degrees(corrected_a)))
    residual_b = abs(float(_estimate_crop_skew_degrees(corrected_b)))

    if residual_a <= residual_b:
        best_image, best_residual = corrected_a, residual_a
    else:
        best_image, best_residual = corrected_b, residual_b

    if best_residual + 0.1 >= original_residual:
        return image, angle, False
    return best_image, angle, True


def _normalize_orientation_mode(orientation_mode):
    raw = str(orientation_mode or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "orizzontale": "horizontal",
        "horizontal": "horizontal",
        "landscape": "horizontal",
        "verticale": "vertical",
        "vertical": "vertical",
        "portrait": "vertical",
    }
    return aliases.get(raw, "auto")


def _enforce_crop_orientation(image, orientation_mode):
    """
    Enforce requested crop orientation.

    Returns:
        (image, normalized_mode, rotated_bool)
    """
    mode = _normalize_orientation_mode(orientation_mode)
    width, height = image.size
    if width <= 0 or height <= 0:
        return image, mode, False

    if mode == "horizontal" and height > width:
        return image.transpose(Image.Transpose.ROTATE_90), mode, True
    if mode == "vertical" and width > height:
        return image.transpose(Image.Transpose.ROTATE_90), mode, True
    return image, mode, False


def split_and_resize_image(
    image_path,
    images_across,
    images_high,
    output_size,
    custom_folder,
    maintain_format,
    smart_grid,
    timestamp,
    output_base_dir=None,
    progress_callback=None,
    crop_margin=0,
    orientation_mode="auto",
):
    """
    Split and resize image, naming crops by short tracking codes (A1, A2, ...).

    Also generates:
    - overlay image with crop boxes + codes
    - CSV index mapping code -> original coordinates
    """
    output_folder = create_output_folder(
        image_path,
        custom_folder,
        images_across,
        images_high,
        timestamp,
        output_base_dir=output_base_dir,
    )
    image_path = Path(image_path)

    crop_margin = int(max(0, crop_margin or 0))
    orientation_mode = _normalize_orientation_mode(orientation_mode)

    with Image.open(image_path) as img:
        img_width, img_height = img.size
        img_format, extension = _resolve_output_format(img, maintain_format)

        crop_boxes = None
        diagnostics = {
            "method": "uniform_fallback",
            "expected_vertical_separators": int(images_across + 1),
            "expected_horizontal_separators": int(images_high + 1),
            "observed_vertical_separators": 0,
            "observed_horizontal_separators": 0,
            "inferred_vertical_separators": int(images_across + 1),
            "inferred_horizontal_separators": int(images_high + 1),
            "offscreen_cells": 0,
            "cell_width_cv": 0.0,
            "cell_height_cv": 0.0,
            "malformed_cell_flags": 0,
            "confidence_score": 0.0,
        }
        if smart_grid and images_across > 1 and images_high > 1:
            crop_boxes, smart_diagnostics = detect_smart_grid_boxes(
                img,
                images_across,
                images_high,
                return_diagnostics=True,
            )
            if crop_boxes is None:
                if not CV_AVAILABLE:
                    print("Warning: Smart grid requested but OpenCV is unavailable. Using classic grid split.")
                else:
                    print(f"Warning: Smart grid fallback for {image_path.name}. Using classic grid split.")
            else:
                diagnostics = smart_diagnostics

        if crop_boxes is None:
            crop_boxes = generate_uniform_grid_boxes(img_width, img_height, images_across, images_high)
            diagnostics = _compute_grid_diagnostics(
                crop_boxes,
                image_size=img.size,
                method="uniform_fallback",
                observed_x_count=0,
                observed_y_count=0,
                expected_x=images_across + 1,
                expected_y=images_high + 1,
            )

        effective_crop_boxes = [_expand_box_with_margin(crop_box, crop_margin) for crop_box in crop_boxes]
        overlap_guard_active = _should_enable_overlap_guard(crop_margin, crop_boxes)
        overlap_corrected_pairs = 0
        max_overlap_before_px = 0
        remaining_overlap_pairs = 0
        remaining_overlap_max_px = 0
        if overlap_guard_active:
            (
                effective_crop_boxes,
                overlap_corrected_pairs,
                max_overlap_before_px,
                remaining_overlap_pairs,
                remaining_overlap_max_px,
            ) = _resolve_adjacent_box_overlaps(
                effective_crop_boxes,
                images_across=images_across,
                images_high=images_high,
                min_side=8,
                max_passes=5,
            )

        codes = _generate_crop_codes(images_across, images_high, len(effective_crop_boxes))
        total_parts = len(effective_crop_boxes)
        straightened_count = 0
        straightened_angles = []
        white_border_trimmed_crops = 0
        max_white_border_trim_px = 0
        white_border_residual_crops = 0
        orientation_rotated_crops = 0

        for idx, code in enumerate(codes):
            crop_box = effective_crop_boxes[idx]
            small_img = _crop_with_white_padding(img, crop_box)
            small_img, estimated_angle, straightened = _auto_straighten_crop(small_img)
            if straightened:
                straightened_count += 1
                straightened_angles.append(float(estimated_angle))

            if crop_margin == 0 and _has_strong_white_border(small_img):
                small_img, trims, trimmed = _trim_white_borders_from_crop(small_img)
                if trimmed:
                    trim_left, trim_top, trim_right, trim_bottom = trims
                    left, top, right, bottom = effective_crop_boxes[idx]
                    effective_crop_boxes[idx] = (
                        int(left + trim_left),
                        int(top + trim_top),
                        int(right - trim_right),
                        int(bottom - trim_bottom),
                    )
                    white_border_trimmed_crops += 1
                    max_white_border_trim_px = max(
                        max_white_border_trim_px,
                        int(trim_left),
                        int(trim_top),
                        int(trim_right),
                        int(trim_bottom),
                    )
                if _has_strong_white_border(small_img):
                    white_border_residual_crops += 1

            small_img, _, orientation_rotated = _enforce_crop_orientation(small_img, orientation_mode)
            if orientation_rotated:
                orientation_rotated_crops += 1

            if output_size is not None and int(output_size) > 0:
                small_img = resize_image_keep_aspect_ratio(small_img, int(output_size))

            output_name = f"{code}.{extension}"
            save_kwargs = {}
            if img_format == "JPEG":
                save_kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
            elif img_format == "WEBP":
                save_kwargs = {"quality": 95, "method": 6}
            small_img.save(output_folder / output_name, img_format, **save_kwargs)

            if progress_callback:
                progress_callback(idx + 1, total_parts)

        _save_tracking_overlay(img, output_folder, image_path.stem, codes, effective_crop_boxes)
        _save_tracking_index(output_folder, image_path.stem, codes, effective_crop_boxes, extension)
        diagnostics["crop_margin_px"] = int(crop_margin)
        diagnostics["auto_straightened_crops"] = int(straightened_count)
        diagnostics["max_abs_straightening_deg"] = round(
            max([abs(val) for val in straightened_angles], default=0.0),
            4,
        )
        diagnostics["white_border_trimmed_crops"] = int(white_border_trimmed_crops)
        diagnostics["max_white_border_trim_px"] = int(max_white_border_trim_px)
        diagnostics["white_border_residual_crops"] = int(white_border_residual_crops)
        diagnostics["overlap_guard_active"] = bool(overlap_guard_active)
        diagnostics["overlap_corrected_pairs"] = int(overlap_corrected_pairs)
        diagnostics["max_overlap_before_correction_px"] = int(max_overlap_before_px)
        diagnostics["remaining_overlap_pairs"] = int(remaining_overlap_pairs)
        diagnostics["remaining_max_overlap_px"] = int(remaining_overlap_max_px)
        diagnostics["orientation_mode"] = str(orientation_mode)
        diagnostics["orientation_rotated_crops"] = int(orientation_rotated_crops)
        _save_diagnostics(output_folder, image_path.stem, diagnostics, total_parts, smart_grid_requested=smart_grid)

    return output_folder


def assess_image_croppability(image_path, images_across, images_high, smart_grid=True, min_confidence=12.0):
    """
    Assess if an image is likely impossible to crop reliably.

    Returns:
        (is_croppable, reason, diagnostics_dict_or_none)
    """
    if images_across <= 0 or images_high <= 0:
        return False, "invalid_grid", None

    if not smart_grid or images_across <= 1 or images_high <= 1:
        return True, "classic_or_single_cell", None

    try:
        with Image.open(image_path) as img:
            boxes, diagnostics = detect_smart_grid_boxes(
                img,
                images_across=images_across,
                images_high=images_high,
                return_diagnostics=True,
            )
    except Exception as exc:
        return False, f"read_error:{exc}", None

    if boxes is None:
        return False, "grid_not_detected", diagnostics

    confidence = float((diagnostics or {}).get("confidence_score", 0.0))
    method = str((diagnostics or {}).get("method", "unknown"))
    observed_v = int((diagnostics or {}).get("observed_vertical_separators", 0))
    observed_h = int((diagnostics or {}).get("observed_horizontal_separators", 0))
    expected_v = int(images_across + 1)
    expected_h = int(images_high + 1)

    if method == "fitted_regular_grid":
        observed_ratio = 0.5 * (
            (observed_v / max(1.0, expected_v)) + (observed_h / max(1.0, expected_h))
        )
        # If almost all separators were inferred (very weak signal), treat as impossible.
        if observed_ratio < 0.45:
            return False, "too_few_detected_lines", diagnostics
        # Keep a strict low-confidence guard without assuming uniform cell sizes.
        if confidence < max(3.0, float(min_confidence) * 0.45) and observed_ratio < 0.60:
            return False, "very_low_confidence", diagnostics

    return True, "ok", diagnostics


def _get_unique_destination_path(destination_folder, source_name):
    destination_folder = Path(destination_folder)
    destination = destination_folder / source_name
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 2
    while True:
        candidate = destination_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _write_impossible_report_csv(impossible_jobs, report_path):
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_path",
                "archived_path",
                "reason",
                "images_across",
                "images_high",
                "smart_grid",
                "method",
                "confidence_score",
            ]
        )
        for job in impossible_jobs:
            diagnostics = job.get("diagnostics") or {}
            writer.writerow(
                [
                    str(job.get("image_path", "")),
                    str(job.get("archived_path", "")),
                    str(job.get("reason", "")),
                    int(job.get("images_across", 0)),
                    int(job.get("images_high", 0)),
                    bool(job.get("smart_grid", False)),
                    diagnostics.get("method", ""),
                    diagnostics.get("confidence_score", ""),
                ]
            )


def assess_and_archive_impossible_files(image_jobs, archive_folder, min_confidence=12.0):
    """
    Assess each job and archive files that are impossible to crop reliably.

    Args:
        image_jobs: iterable of dicts with keys:
            - image_path
            - images_across
            - images_high
            - smart_grid
          Additional keys are preserved in returned entries.
        archive_folder: destination folder for impossible files/report.
        min_confidence: threshold passed to assess_image_croppability.

    Returns:
        (processable_jobs, impossible_jobs, report_path_or_none)
    """
    archive_folder = Path(archive_folder)
    processable = []
    impossible = []

    for raw_job in image_jobs:
        job = dict(raw_job)
        image_path = job.get("image_path")
        images_across = int(job.get("images_across", 0))
        images_high = int(job.get("images_high", 0))
        smart_grid = bool(job.get("smart_grid", True))

        is_ok, reason, diagnostics = assess_image_croppability(
            image_path=image_path,
            images_across=images_across,
            images_high=images_high,
            smart_grid=smart_grid,
            min_confidence=min_confidence,
        )
        if is_ok:
            processable.append(job)
            continue

        archive_folder.mkdir(parents=True, exist_ok=True)
        archived_path = None
        source_path = Path(image_path)
        if source_path.exists():
            try:
                destination = _get_unique_destination_path(archive_folder, source_path.name)
                shutil.copy2(source_path, destination)
                archived_path = str(destination)
            except Exception as exc:
                reason = f"{reason};archive_error:{exc}"

        job["reason"] = reason
        job["diagnostics"] = diagnostics
        job["archived_path"] = archived_path
        impossible.append(job)

    report_path = None
    if impossible:
        archive_folder.mkdir(parents=True, exist_ok=True)
        report_path = archive_folder / "impossible_report.csv"
        _write_impossible_report_csv(impossible, report_path)

    return processable, impossible, report_path
