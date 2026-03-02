import sys
import argparse
import datetime
import threading
import json
from pathlib import Path
from tkinter import filedialog, StringVar, IntVar, BooleanVar, messagebox, Menu, Toplevel
from tkinter import ttk
from tkinter import scrolledtext
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from .splitter_models import ImageItem, ImageSplitterConfig
from .splitter_core import (
    CV_AVAILABLE,
    is_cv_available,
    create_output_folder,
    detect_smart_grid_boxes,
    split_and_resize_image,
    assess_image_croppability,
    assess_and_archive_impossible_files,
    _index_to_alpha,
    _generate_crop_codes,
    _crop_with_white_padding,
)

try:
    import sv_ttk
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

TOOLTIP_CONFIG_FILE = Path(__file__).resolve().parent.parent / "tooltips.json"

HELP_TEXT = (
    "Splitter - User Help Guide\n"
    "\n"
    "Splitter divides scanned sheets into smaller images using configurable grid settings.\n"
    "This guide explains the interface and the full processing flow.\n"
    "\n"
    "Getting Started\n"
    "1. Load image files.\n"
    "2. Configure global defaults.\n"
    "3. Optionally configure per-image overrides.\n"
    "4. Start processing.\n"
    "\n"
    "Image List Panel\n"
    "- Displays loaded images.\n"
    "- Add images with Browse or drag-and-drop.\n"
    "- Remove items with Remove Selected or clear all with Clear List.\n"
    "- Selecting an item updates preview and per-image settings.\n"
    "\n"
    "Global Default Settings\n"
    "- Configure grid, orientation, crop margin, and output paths.\n"
    "- Set Output Quality to control final sharpness and file size.\n"
    "- For maximum quality use: Massima (consigliata).\n"
    "- Output Location defines the base directory for generated folders.\n"
    "- Output Folder is an optional subfolder name for exports.\n"
    "\n"
    "Per-Image Settings\n"
    "- Enable Use custom settings for this image to override global defaults.\n"
    "- You can force crop orientation: Auto, Orizzontale, or Verticale.\n"
    "- Click Apply to Image to save overrides for the selected file.\n"
    "- Click Reset to Defaults to remove overrides.\n"
    "\n"
    "Processing Flow\n"
    "- At start, Splitter validates each file using current effective settings.\n"
    "- During crop export, Splitter applies automatic skew control to reduce tilted results.\n"
    "- With margin 0, Splitter runs a post-crop white-border check and trims residual white edges automatically.\n"
    "- With very high margins, Splitter applies an anti-overlap guard to prevent intersections between adjacent crops.\n"
    "- Files that can be processed are split and exported.\n"
    "- Tracking files are generated for each source sheet:\n"
    "  <source>_tracking_overlay.jpg, <source>_tracking_index.csv, <source>_diagnostics.json.\n"
    "\n"
    "Automatic Impossible-File Handling\n"
    "- Impossible files are detected automatically at process start.\n"
    "- Those files are copied to _impossible_files/<timestamp> and skipped.\n"
    "- A report CSV named impossible_report.csv is generated in that folder.\n"
    "- The app shows a warning dialog with summary and reasons.\n"
    "- At the end, a conversion report window lists each file with status, score, and details.\n"
)


def extract_accuracy_score(diagnostics):
    """Extract a normalized 0..100 confidence score from diagnostics."""
    if not isinstance(diagnostics, dict):
        return None
    raw = diagnostics.get("confidence_score")
    if raw is None or raw == "":
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, score))


def classify_accuracy_band(score):
    """Classify numeric score into a human-readable band."""
    if score is None:
        return "n/a"
    if float(score) >= 85.0:
        return "Alta"
    if float(score) >= 60.0:
        return "Media"
    return "Bassa"


def format_accuracy_score(score):
    """Format score for UI display."""
    if score is None:
        return "-"
    return f"{float(score):.1f} ({classify_accuracy_band(score)})"


def build_diagnostic_summary(diagnostics):
    """Build a compact diagnostics summary string for report rows."""
    if not isinstance(diagnostics, dict) or not diagnostics:
        return "No diagnostics"

    parts = []
    method = diagnostics.get("method")
    if method:
        parts.append(f"method={method}")
    orientation_mode = diagnostics.get("orientation_mode")
    if orientation_mode:
        parts.append(f"orientation={orientation_mode}")

    malformed = int(diagnostics.get("malformed_cell_flags", 0) or 0)
    offscreen = int(diagnostics.get("offscreen_cells", 0) or 0)
    if malformed > 0:
        parts.append(f"malformed={malformed}")
    if offscreen > 0:
        parts.append(f"offscreen={offscreen}")

    trimmed = int(diagnostics.get("white_border_trimmed_crops", 0) or 0)
    overlap_fixed = int(diagnostics.get("overlap_corrected_pairs", 0) or 0)
    overlap_left = int(diagnostics.get("remaining_overlap_pairs", 0) or 0)
    if trimmed > 0:
        parts.append(f"white_trim={trimmed}")
    if overlap_fixed > 0:
        parts.append(f"overlap_fixed={overlap_fixed}")
    if overlap_left > 0:
        parts.append(f"overlap_left={overlap_left}")

    inferred_v = int(diagnostics.get("inferred_vertical_separators", 0) or 0)
    inferred_h = int(diagnostics.get("inferred_horizontal_separators", 0) or 0)
    if inferred_v > 0 or inferred_h > 0:
        parts.append(f"inferred=v{inferred_v}/h{inferred_h}")

    return ", ".join(parts) if parts else "Diagnostics available"


class ImageSplitterGUI:
    """Modern GUI for image splitting application."""
    
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    PREVIEW_SIZE = 300
    QUALITY_MAX = "Massima (consigliata)"
    QUALITY_BALANCED = "Bilanciata"
    QUALITY_COMPACT = "Compatta"
    QUALITY_TO_SIZE = {
        QUALITY_MAX: None,       # Keep native crop resolution
        QUALITY_BALANCED: 2048,  # Downscale to reduce output weight
        QUALITY_COMPACT: 1024,   # Smaller files for quick usage
    }
    ORIENTATION_AUTO = "Auto"
    ORIENTATION_HORIZONTAL = "Orizzontale"
    ORIENTATION_VERTICAL = "Verticale"
    ORIENTATION_TO_VALUE = {
        ORIENTATION_AUTO: "auto",
        ORIENTATION_HORIZONTAL: "horizontal",
        ORIENTATION_VERTICAL: "vertical",
    }
    VALUE_TO_ORIENTATION = {
        "auto": ORIENTATION_AUTO,
        "horizontal": ORIENTATION_HORIZONTAL,
        "vertical": ORIENTATION_VERTICAL,
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Splitter and Resizer - Per-Image Settings")
        self.root.minsize(900, 600)
        
        # Apply modern theme
        if THEME_AVAILABLE:
            sv_ttk.set_theme("dark")
        
        self.config = ImageSplitterConfig()
        self.preview_image = None
        self.tooltips = {}
        self.load_tooltips()
        self.create_menubar()
        self.setup_variables()
        self.setup_ui()
        self.validate_inputs()
        if not is_cv_available():
            self.status_var.set("Warning: OpenCV unavailable. Smart grid will use classic split fallback.")

    def load_tooltips(self):
        """Load tooltip overrides from JSON configuration file."""
        if TOOLTIP_CONFIG_FILE.exists():
            try:
                # utf-8-sig handles optional BOM from editors on Windows.
                with TOOLTIP_CONFIG_FILE.open("r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self.tooltips = {str(key): str(value) for key, value in data.items()}
                else:
                    print("Warning: tooltips.json must contain a JSON object of key/value pairs.")
                    self.tooltips = {}
            except Exception as exc:
                print(f"Warning: Failed to load tooltip configuration: {exc}")
                self.tooltips = {}
        else:
            self.tooltips = {}
        
    def create_menubar(self):
        """Create application menu bar."""
        menubar = Menu(self.root)

        file_menu = Menu(menubar, tearoff=False)
        file_menu.add_command(label="Reset", command=self.reset_application)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(menubar, tearoff=False)
        help_menu.add_command(label="User Guide", command=self.show_help_manual)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def reset_application(self):
        """Reset all settings and clear the workspace."""
        if not messagebox.askyesno(
            "Reset Application",
            "Clear all images and restore the default settings?"
        ):
            return

        # Reset global defaults
        self.global_quality_mode_var.set(self.QUALITY_MAX)
        self.global_orientation_mode_var.set(self.ORIENTATION_AUTO)
        self.global_images_across_var.set(1)
        self.global_images_high_var.set(1)
        self.global_crop_margin_var.set("0")
        self.global_folder_name_var.set("")
        self.global_output_base_dir_var.set("")

        # Reset per-image defaults
        self.images_across_var.set(1)
        self.images_high_var.set(1)
        self.orientation_mode_var.set(self.global_orientation_mode_var.get())
        self.crop_margin_var.set("0")
        self.folder_name_var.set("")
        self.use_custom_settings_var.set(False)
        self.toggle_settings_state(False)

        # Clear images and preview
        self.clear_list()
        self.status_var.set("Restored default settings.")
        self.validate_inputs()

    def show_help_manual(self):
        """Display the help guide in a separate window."""
        help_window = Toplevel(self.root)
        help_window.title("Splitter - User Help Guide")
        help_window.geometry("720x640")
        help_window.transient(self.root)
        help_window.grab_set()

        text_area = scrolledtext.ScrolledText(help_window, wrap='word')
        text_area.pack(fill='both', expand=True)
        text_area.insert('1.0', HELP_TEXT)
        text_area.configure(state='disabled')
        text_area.focus_set()

        def close_on_escape(event=None):
            help_window.destroy()

        help_window.bind('<Escape>', close_on_escape)

    def setup_variables(self):
        """Initialize tkinter variables."""
        # Global defaults
        self.global_quality_mode_var = StringVar(value=self.QUALITY_MAX)
        self.global_orientation_mode_var = StringVar(value=self.ORIENTATION_AUTO)
        self.global_folder_name_var = StringVar()
        self.global_output_base_dir_var = StringVar()
        self.global_crop_margin_var = StringVar(value="0")
        self.global_images_across_var = IntVar(value=1)
        self.global_images_high_var = IntVar(value=1)
        
        # Per-image settings
        self.folder_name_var = StringVar()
        self.crop_margin_var = StringVar(value="0")
        self.images_across_var = IntVar(value=1)
        self.images_high_var = IntVar(value=1)
        self.orientation_mode_var = StringVar(value=self.ORIENTATION_AUTO)
        self.use_custom_settings_var = BooleanVar(value=False)
        
        self.status_var = StringVar(value="Ready")
        
        # Add validation traces
        self.global_crop_margin_var.trace('w', lambda *args: self.validate_inputs())
        self.global_images_across_var.trace('w', lambda *args: self.validate_inputs())
        self.global_images_high_var.trace('w', lambda *args: self.validate_inputs())
        self.crop_margin_var.trace('w', lambda *args: self.validate_inputs())
        self.images_across_var.trace('w', lambda *args: self.validate_inputs())
        self.images_high_var.trace('w', lambda *args: self.validate_inputs())
        self.use_custom_settings_var.trace('w', lambda *args: self.on_custom_settings_toggle())
        
    def setup_ui(self):
        """Create the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky='nsew')
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create paned window for resizable sections
        paned = ttk.PanedWindow(main_frame, orient='horizontal')
        paned.grid(row=0, column=0, sticky='nsew')
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Left panel - File list and global settings
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=0)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Right panel - Preview and per-image settings
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=1)
        right_panel.grid_rowconfigure(0, weight=3)
        right_panel.grid_rowconfigure(1, weight=2)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Setup left panel
        self.create_file_section(left_panel)
        self.create_global_settings_section(left_panel)
        
        # Setup right panel
        self.create_preview_section(right_panel)
        self.create_per_image_settings_section(right_panel)
        
        # Action buttons and status at bottom
        self.create_action_buttons(main_frame)
        self.create_status_section(main_frame)
        
    def create_file_section(self, parent):
        """Create file selection UI."""
        file_frame = ttk.LabelFrame(parent, text="Image Files", padding="10")
        file_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        file_frame.grid_rowconfigure(1, weight=1)
        file_frame.grid_columnconfigure(0, weight=1)
        
        # Browse button
        btn_frame = ttk.Frame(file_frame)
        btn_frame.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        self.browse_btn = ttk.Button(btn_frame, text="Browse Images", command=self.browse_images)
        self.browse_btn.pack(side='left', padx=(0, 5))
        self.create_tooltip(self.browse_btn, "browse_button", "Select one or more image files to process")
        
        ttk.Label(btn_frame, text="or drag and drop files below", foreground="gray").pack(side='left')
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.grid(row=1, column=0, sticky='nsew')
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Use Treeview with columns for better display
        self.listbox = ttk.Treeview(
            list_frame,
            yscrollcommand=scrollbar.set,
            columns=('settings',),
            show='tree headings',
            height=10,
        )
        self.listbox.grid(row=0, column=0, sticky='nsew')
        self.listbox.heading('#0', text='File')
        self.listbox.heading('settings', text='Settings')
        self.listbox.column('#0', width=300, minwidth=180, stretch=True)
        self.listbox.column('settings', width=220, minwidth=140, stretch=True)
        scrollbar.config(command=self.listbox.yview)
        self.create_tooltip(self.listbox, "image_list", "Select an image to preview and adjust settings")
        self.listbox.bind('<Configure>', self.on_listbox_resize)
        
        # Bind selection event
        self.listbox.bind('<<TreeviewSelect>>', self.on_image_select)
        
        # Enable drag and drop
        self.listbox.drop_target_register(DND_FILES)
        self.listbox.dnd_bind('<<Drop>>', self.on_drop)
        
        # Buttons
        btn_frame2 = ttk.Frame(file_frame)
        btn_frame2.grid(row=2, column=0, sticky='ew', pady=(5, 0))
        
        self.clear_btn = ttk.Button(btn_frame2, text="Clear List", command=self.clear_list)
        self.clear_btn.pack(side='left', padx=(0, 5))
        self.create_tooltip(self.clear_btn, "clear_list_button", "Remove all images from the list")
        
        self.remove_btn = ttk.Button(btn_frame2, text="Remove Selected", command=self.remove_selected)
        self.remove_btn.pack(side='left', padx=(0, 5))
        self.create_tooltip(self.remove_btn, "remove_selected_button", "Remove the highlighted image from the list")
        
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
    def create_global_settings_section(self, parent):
        """Create global default settings UI."""
        settings_frame = ttk.LabelFrame(parent, text="Global Default Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))

        ttk.Label(settings_frame, text="Output Quality:").grid(row=0, column=0, sticky='w')
        quality_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.global_quality_mode_var,
            values=[self.QUALITY_MAX, self.QUALITY_BALANCED, self.QUALITY_COMPACT],
            state='readonly',
            width=24,
        )
        quality_combo.grid(row=0, column=1, sticky='w')
        self.create_tooltip(
            quality_combo,
            "global_quality_mode",
            "Choose output quality. Massima keeps full crop resolution for best detail."
        )

        self.quality_hint_label = ttk.Label(
            settings_frame,
            text="Suggerimento: usa 'Massima (consigliata)' per massimizzare la qualità.",
            foreground="gray",
            wraplength=420,
        )
        self.quality_hint_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 6))
        settings_frame.bind('<Configure>', self.on_global_settings_resize)

        ttk.Label(settings_frame, text="Orientation:").grid(row=2, column=0, sticky='w')
        orientation_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.global_orientation_mode_var,
            values=[self.ORIENTATION_AUTO, self.ORIENTATION_HORIZONTAL, self.ORIENTATION_VERTICAL],
            state='readonly',
            width=24,
        )
        orientation_combo.grid(row=2, column=1, sticky='w')
        self.create_tooltip(
            orientation_combo,
            "global_orientation_mode",
            "Force exported crop orientation: Auto, Orizzontale, or Verticale."
        )

        # Grid options
        grid_frame = ttk.Frame(settings_frame)
        grid_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        
        ttk.Label(grid_frame, text="Across:").pack(side='left', padx=(0, 5))
        global_across_spin = ttk.Spinbox(grid_frame, from_=1, to=10, textvariable=self.global_images_across_var, width=8)
        global_across_spin.pack(side='left', padx=(0, 10))
        self.create_tooltip(global_across_spin, "global_images_across", "Number of columns to split each image into")
        
        ttk.Label(grid_frame, text="High:").pack(side='left', padx=(0, 5))
        global_high_spin = ttk.Spinbox(grid_frame, from_=1, to=10, textvariable=self.global_images_high_var, width=8)
        global_high_spin.pack(side='left')
        self.create_tooltip(global_high_spin, "global_images_high", "Number of rows to split each image into")
        
        ttk.Label(settings_frame, text="Crop Margin (px):").grid(row=4, column=0, sticky='w', pady=(5, 0))
        self.global_crop_margin_entry = ttk.Entry(settings_frame, textvariable=self.global_crop_margin_var, width=12)
        self.global_crop_margin_entry.grid(row=4, column=1, sticky='w', pady=(5, 0))
        self.create_tooltip(
            self.global_crop_margin_entry,
            "global_crop_margin",
            "Extra pixels added around each detected rectangle. Use 0 for no margin."
        )

        ttk.Label(settings_frame, text="Output Folder:").grid(row=5, column=0, sticky='w', pady=(5, 0))
        global_folder_entry = ttk.Entry(settings_frame, textvariable=self.global_folder_name_var)
        global_folder_entry.grid(row=5, column=1, sticky='ew', pady=(5, 0))
        self.create_tooltip(global_folder_entry, "global_output_folder", "Optional subfolder name created inside each image's directory")

        ttk.Label(settings_frame, text="Output Location:").grid(row=6, column=0, sticky='w', pady=(5, 0))
        output_location_frame = ttk.Frame(settings_frame)
        output_location_frame.grid(row=6, column=1, sticky='ew', pady=(5, 0))
        output_location_frame.grid_columnconfigure(0, weight=1)

        output_location_entry = ttk.Entry(output_location_frame, textvariable=self.global_output_base_dir_var)
        output_location_entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.create_tooltip(
            output_location_entry,
            "global_output_location",
            "Base directory where all output folders will be created. Leave empty to use source image folders."
        )

        output_location_btn = ttk.Button(output_location_frame, text="Browse...", command=self.browse_output_base_directory)
        output_location_btn.grid(row=0, column=1, sticky='e')
        self.create_tooltip(output_location_btn, "global_output_location_browse", "Choose output base directory")
        settings_frame.grid_columnconfigure(1, weight=1)
        
    def create_preview_section(self, parent):
        """Create image preview UI."""
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding="10")
        preview_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        # Info label
        self.preview_info_label = ttk.Label(preview_frame, text="Select an image to preview")
        self.preview_info_label.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        # Preview canvas
        preview_canvas_frame = ttk.Frame(preview_frame, relief='sunken', borderwidth=2)
        preview_canvas_frame.grid(row=1, column=0, sticky='nsew')
        preview_canvas_frame.grid_rowconfigure(0, weight=1)
        preview_canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.preview_label = ttk.Label(preview_canvas_frame, text="No image selected", 
                                       anchor='center', background='#2b2b2b')
        self.preview_label.grid(row=0, column=0, sticky='nsew')
        
    def create_per_image_settings_section(self, parent):
        """Create per-image settings UI."""
        settings_frame = ttk.LabelFrame(parent, text="Image-Specific Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky='nsew')
        settings_frame.grid_columnconfigure(0, weight=1)
        
        # Enable custom settings checkbox
        self.use_custom_check = ttk.Checkbutton(settings_frame, 
                                               text="Use custom settings for this image",
                                               variable=self.use_custom_settings_var)
        self.use_custom_check.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))
        self.create_tooltip(self.use_custom_check, "per_image_enable_custom", "Override global defaults for the selected image")
        
        # Settings container
        self.settings_container = ttk.Frame(settings_frame)
        self.settings_container.grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.settings_container.grid_columnconfigure(1, weight=1)
        
        row = 0
        ttk.Label(self.settings_container, text="Images Across:").grid(row=row, column=0, sticky='w', pady=2)
        self.across_entry = ttk.Spinbox(self.settings_container, from_=1, to=10, 
                                       textvariable=self.images_across_var, width=15)
        self.across_entry.grid(row=row, column=1, sticky='w', pady=2)
        self.create_tooltip(self.across_entry, "per_image_images_across", "Number of columns to slice for this image")
        
        row += 1
        ttk.Label(self.settings_container, text="Images High:").grid(row=row, column=0, sticky='w', pady=2)
        self.high_entry = ttk.Spinbox(self.settings_container, from_=1, to=10, 
                                     textvariable=self.images_high_var, width=15)
        self.high_entry.grid(row=row, column=1, sticky='w', pady=2)
        self.create_tooltip(self.high_entry, "per_image_images_high", "Number of rows to slice for this image")

        row += 1
        ttk.Label(self.settings_container, text="Orientation:").grid(row=row, column=0, sticky='w', pady=2)
        self.orientation_combo = ttk.Combobox(
            self.settings_container,
            textvariable=self.orientation_mode_var,
            values=[self.ORIENTATION_AUTO, self.ORIENTATION_HORIZONTAL, self.ORIENTATION_VERTICAL],
            state='readonly',
            width=15,
        )
        self.orientation_combo.grid(row=row, column=1, sticky='w', pady=2)
        self.create_tooltip(
            self.orientation_combo,
            "per_image_orientation_mode",
            "Force crop orientation for this image only: Auto, Orizzontale, or Verticale."
        )

        row += 1
        ttk.Label(self.settings_container, text="Crop Margin (px):").grid(row=row, column=0, sticky='w', pady=2)
        self.crop_margin_entry = ttk.Entry(self.settings_container, textvariable=self.crop_margin_var, width=15)
        self.crop_margin_entry.grid(row=row, column=1, sticky='w', pady=2)
        self.create_tooltip(
            self.crop_margin_entry,
            "per_image_crop_margin",
            "Extra pixels added around each crop for this image. Use 0 for no margin."
        )

        row += 1
        ttk.Label(self.settings_container, text="Output Folder:").grid(row=row, column=0, sticky='w', pady=2)
        per_image_folder_entry = ttk.Entry(self.settings_container, textvariable=self.folder_name_var)
        per_image_folder_entry.grid(row=row, column=1, sticky='ew', pady=2)
        self.create_tooltip(per_image_folder_entry, "per_image_output_folder", "Optional subfolder name for this image's output")
        
        # Apply/Reset buttons
        row += 1
        btn_frame = ttk.Frame(self.settings_container)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        
        self.apply_btn = ttk.Button(btn_frame, text="Apply to Image", command=self.apply_settings)
        self.apply_btn.pack(side='left', padx=(0, 5))
        self.create_tooltip(self.apply_btn, "per_image_apply", "Save the current settings to the selected image")
        
        self.reset_btn = ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_to_defaults)
        self.reset_btn.pack(side='left')
        self.create_tooltip(self.reset_btn, "per_image_reset", "Remove custom settings and use global defaults")
        
        # Initially disable settings
        self.toggle_settings_state(False)
        
    def create_action_buttons(self, parent):
        """Create action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        self.process_btn = ttk.Button(button_frame, text="Process All Images", 
                                      command=self.start_processing, style='Accent.TButton')
        self.process_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.create_tooltip(self.process_btn, "process_all_button", "Begin processing all images using their configured settings")
        
        self.cancel_btn = ttk.Button(button_frame, text="Cancel", 
                                     command=self.cancel_processing, state='disabled')
        self.cancel_btn.grid(row=0, column=1, sticky='ew')
        self.create_tooltip(self.cancel_btn, "cancel_button", "Stop the current processing job")
        
    def create_status_section(self, parent):
        """Create status and progress display."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, sticky='w')

    def create_tooltip(self, widget, key, default_text=""):
        """Create tooltip for widget using JSON overrides when available."""
        tooltip_text = self.tooltips.get(key, default_text)
        if not tooltip_text:
            return

        def show_tooltip(event):
            tooltip = ttk.Label(self.root, text=tooltip_text, relief='solid', borderwidth=1, 
                               background='#ffffe0', foreground='#000000', padding=5)
            tooltip.update_idletasks()

            self.root.update_idletasks()
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_width = self.root.winfo_width()
            root_height = self.root.winfo_height()
            tip_width = tooltip.winfo_reqwidth()
            tip_height = tooltip.winfo_reqheight()

            desired_x = event.x_root - root_x + 10
            desired_y = event.y_root - root_y + 10

            max_x = max(0, root_width - tip_width - 10)
            max_y = max(0, root_height - tip_height - 10)

            clamped_x = min(max(desired_x, 0), max_x)
            clamped_y = min(max(desired_y, 0), max_y)

            tooltip.place(x=clamped_x, y=clamped_y)
            widget.tooltip = tooltip

        def on_enter(event):
            def delayed_show():
                show_tooltip(event)

            if hasattr(widget, 'tooltip_after_id'):
                self.root.after_cancel(widget.tooltip_after_id)
            widget.tooltip_after_id = self.root.after(1000, delayed_show)

        def on_leave(event):
            if hasattr(widget, 'tooltip_after_id'):
                self.root.after_cancel(widget.tooltip_after_id)
                delattr(widget, 'tooltip_after_id')
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
        
    def toggle_settings_state(self, enabled):
        """Enable or disable per-image settings widgets."""
        state = 'normal' if enabled else 'disabled'
        for child in self.settings_container.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.Spinbox, ttk.Combobox, ttk.Checkbutton, ttk.Button)):
                child.configure(state=state)

    def on_listbox_resize(self, event):
        """Keep Treeview columns balanced when the window is resized."""
        total_width = max(1, int(event.width))
        file_col = max(180, int(total_width * 0.58))
        settings_col = max(140, total_width - file_col - 6)
        self.listbox.column('#0', width=file_col)
        self.listbox.column('settings', width=settings_col)

    def on_global_settings_resize(self, event):
        """Adapt help hint wrapping to panel width to avoid clipping."""
        if hasattr(self, "quality_hint_label"):
            wrap = max(220, int(event.width) - 24)
            self.quality_hint_label.configure(wraplength=wrap)
                
    def on_custom_settings_toggle(self):
        """Handle custom settings checkbox toggle."""
        self.toggle_settings_state(self.use_custom_settings_var.get())
    
    def on_image_select(self, event):
        """Handle image selection in listbox."""
        selection = self.listbox.selection()
        if not selection:
            self.config.selected_item = None
            self.clear_preview()
            return
            
        item_id = selection[0]
        for img_item in self.config.image_items:
            if str(id(img_item)) == item_id:
                self.config.selected_item = img_item
                self.load_preview(img_item)
                self.load_image_settings(img_item)
                break
                
    def load_preview(self, img_item):
        """Load and display image preview."""
        try:
            with Image.open(img_item.file_path) as img:
                width, height = img.size
                file_size = img_item.file_path.stat().st_size / 1024
                
                info_text = f"{img_item.file_path.name} | {width}x{height} | {file_size:.1f} KB"
                self.preview_info_label.config(text=info_text)
                
                img.thumbnail((self.PREVIEW_SIZE, self.PREVIEW_SIZE), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                self.preview_image = photo
                self.preview_label.config(image=photo, text="")
        except Exception as e:
            self.preview_label.config(image='', text=f"Error loading preview:\n{str(e)}")
            self.preview_info_label.config(text="Error loading image")
            
    def clear_preview(self):
        """Clear the preview display."""
        self.preview_label.config(image='', text="No image selected")
        self.preview_info_label.config(text="Select an image to preview")
        self.preview_image = None
        self.use_custom_settings_var.set(False)
        self.toggle_settings_state(False)
        
    def load_image_settings(self, img_item):
        """Load settings for selected image."""
        if img_item.has_custom_settings():
            self.use_custom_settings_var.set(True)
            self.images_across_var.set(img_item.images_across or self.global_images_across_var.get())
            self.images_high_var.set(img_item.images_high or self.global_images_high_var.get())
            self.orientation_mode_var.set(
                self._orientation_label_from_value(
                    img_item.orientation_mode or self._resolve_orientation_mode(self.global_orientation_mode_var.get())
                )
            )
            self.crop_margin_var.set(str(img_item.crop_margin if img_item.crop_margin is not None else self.global_crop_margin_var.get()))
            self.folder_name_var.set(img_item.custom_folder or self.global_folder_name_var.get())
        else:
            self.use_custom_settings_var.set(False)
            self.images_across_var.set(self.global_images_across_var.get())
            self.images_high_var.set(self.global_images_high_var.get())
            self.orientation_mode_var.set(self.global_orientation_mode_var.get())
            self.crop_margin_var.set(self.global_crop_margin_var.get())
            self.folder_name_var.set(self.global_folder_name_var.get())
            
    def apply_settings(self):
        """Apply current settings to selected image."""
        
        if not self.config.selected_item:
            return
            
        if self.use_custom_settings_var.get():
            img_item = self.config.selected_item
            img_item.base_size = None
            img_item.custom_size = None
            img_item.images_across = self.images_across_var.get()
            img_item.images_high = self.images_high_var.get()
            img_item.maintain_format = None
            img_item.smart_grid = None
            img_item.orientation_mode = self._resolve_orientation_mode(self.orientation_mode_var.get())
            margin = self.crop_margin_var.get().strip()
            img_item.crop_margin = int(margin) if margin.isdigit() else None
            
            folder = self.folder_name_var.get().strip()
            img_item.custom_folder = folder if folder else None

            self.update_listbox_item(img_item)
            self.status_var.set(f"Applied custom settings to {img_item.file_path.name}.")
        else:
            self.reset_to_defaults()
            
    def reset_to_defaults(self):
        """Reset selected image to use global defaults."""
        if not self.config.selected_item:
            return
            
        img_item = self.config.selected_item
        img_item.base_size = None
        img_item.custom_size = None
        img_item.images_across = None
        img_item.images_high = None
        img_item.maintain_format = None
        img_item.smart_grid = None
        img_item.orientation_mode = None
        img_item.crop_margin = None
        img_item.custom_folder = None
        
        self.use_custom_settings_var.set(False)
        self.load_image_settings(img_item)
        self.update_listbox_item(img_item)
        self.status_var.set(f"Reset {img_item.file_path.name} to global defaults.")
        
    def update_listbox_item(self, img_item):
        """Update listbox display for an image item."""
        item_id = str(id(img_item))
        if self.listbox.exists(item_id):
            self.listbox.item(item_id, text=img_item.get_display_name(), 
                            values=(img_item.get_settings_summary(),))

    def validate_inputs(self):
        """Validate user inputs and update UI state."""
        valid = True

        # Validate global crop margin
        global_margin = self.global_crop_margin_var.get().strip()
        if global_margin and not global_margin.isdigit():
            valid = False
            if hasattr(self, "global_crop_margin_entry"):
                self.global_crop_margin_entry.state(['invalid'])
        else:
            if hasattr(self, "global_crop_margin_entry"):
                self.global_crop_margin_entry.state(['!invalid'])

        # Validate per-image crop margin
        per_image_margin = self.crop_margin_var.get().strip()
        if per_image_margin and not per_image_margin.isdigit():
            valid = False
            if hasattr(self, "crop_margin_entry"):
                self.crop_margin_entry.state(['invalid'])
        else:
            if hasattr(self, "crop_margin_entry"):
                self.crop_margin_entry.state(['!invalid'])
            
        # Validate grid dimensions
        try:
            across = self.images_across_var.get()
            high = self.images_high_var.get()
            global_across = self.global_images_across_var.get()
            global_high = self.global_images_high_var.get()
            if across <= 0 or high <= 0 or global_across <= 0 or global_high <= 0:
                valid = False
        except:
            valid = False
            
        # Enable/disable process button
        has_images = len(self.config.image_items) > 0
        if valid and has_images and not self.config.processing:
            self.process_btn.state(['!disabled'])
        else:
            self.process_btn.state(['disabled'])
            
        return valid

    def browse_output_base_directory(self):
        """Select a base output directory via file explorer."""
        initial_dir = self.global_output_base_dir_var.get().strip() or str(Path.cwd())
        selected_dir = filedialog.askdirectory(title="Select Output Base Directory", initialdir=initial_dir)
        if selected_dir:
            self.global_output_base_dir_var.set(selected_dir)
            self.status_var.set(f"Output location set to: {selected_dir}")

    def _resolve_output_size(self):
        mode = self.global_quality_mode_var.get()
        return self.QUALITY_TO_SIZE.get(mode, None)

    def _resolve_orientation_mode(self, label):
        return self.ORIENTATION_TO_VALUE.get(label, "auto")

    def _orientation_label_from_value(self, value):
        normalized = str(value or "auto").strip().lower()
        return self.VALUE_TO_ORIENTATION.get(normalized, self.ORIENTATION_AUTO)

    def get_effective_image_settings(self, img_item):
        """Resolve effective settings for one image (global + per-image overrides)."""
        output_size = self._resolve_output_size()
        global_orientation = self._resolve_orientation_mode(self.global_orientation_mode_var.get())
        global_margin_text = self.global_crop_margin_var.get().strip()
        global_margin = int(global_margin_text) if global_margin_text.isdigit() else 0
        crop_margin = img_item.crop_margin if img_item.crop_margin is not None else global_margin
        orientation_mode = img_item.orientation_mode if img_item.orientation_mode is not None else global_orientation

        return {
            "output_size": output_size,
            "across": img_item.images_across or self.global_images_across_var.get(),
            "high": img_item.images_high or self.global_images_high_var.get(),
            "orientation_mode": orientation_mode,
            "folder": img_item.custom_folder or self.global_folder_name_var.get(),
            "format_setting": False,
            "smart_grid_setting": True,
            "crop_margin": crop_margin,
            "output_base_dir": self.global_output_base_dir_var.get().strip() or None,
        }

    def _resolve_impossible_archive_dir(self, timestamp):
        base_output = self.global_output_base_dir_var.get().strip()
        if base_output:
            base_dir = Path(base_output)
        elif self.config.image_items:
            base_dir = self.config.image_items[0].file_path.parent
        else:
            base_dir = Path.cwd()
        global_folder = self.global_folder_name_var.get().strip()
        if global_folder:
            base_dir = base_dir / global_folder
        return base_dir / "_impossible_files" / timestamp

    def _build_processing_jobs(self):
        jobs = []
        for img_item in self.config.image_items:
            settings = self.get_effective_image_settings(img_item)
            jobs.append(
                {
                    "image_item": img_item,
                    "item_id": str(id(img_item)),
                    "image_path": str(img_item.file_path),
                    "images_across": settings["across"],
                    "images_high": settings["high"],
                    "smart_grid": settings["smart_grid_setting"],
                    "settings": settings,
                }
            )
        return jobs

    def _highlight_impossible_items(self, impossible_jobs):
        for current_selection in self.listbox.selection():
            self.listbox.selection_remove(current_selection)

        impossible_ids = [job.get("item_id") for job in impossible_jobs if job.get("item_id")]
        if not impossible_ids:
            return

        self.listbox.selection_set(impossible_ids)
        self.listbox.focus(impossible_ids[0])
        self.listbox.see(impossible_ids[0])

    def _notify_impossible_jobs(self, impossible_jobs, archive_dir, report_path):
        if not impossible_jobs:
            return

        preview_lines = []
        for job in impossible_jobs[:10]:
            name = Path(job.get("image_path", "")).name
            reason = job.get("reason", "unknown")
            method = (job.get("diagnostics") or {}).get("method", "n/a")
            score = (job.get("diagnostics") or {}).get("confidence_score", "n/a")
            preview_lines.append(f"- {name}: {reason} (method={method}, score={score})")
        preview = "\n".join(preview_lines)
        if len(impossible_jobs) > 10:
            preview += f"\n... and {len(impossible_jobs) - 10} more."

        report_line = f"\nReport: {report_path}" if report_path else ""
        messagebox.showwarning(
            "Impossible Files Detected",
            (
                f"{len(impossible_jobs)} file(s) were detected as impossible to crop reliably.\n"
                f"They were copied to:\n{archive_dir}{report_line}\n\n{preview}"
            ),
        )

    def _load_conversion_diagnostics(self, output_folder, source_stem):
        diagnostics_path = Path(output_folder) / f"{source_stem}_diagnostics.json"
        if not diagnostics_path.exists():
            return {}
        try:
            with diagnostics_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _build_conversion_row(self, file_name, status, diagnostics=None, reason="", error="", output_path=""):
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        score = extract_accuracy_score(diagnostics)
        method = diagnostics.get("method") if diagnostics else None

        detail_parts = []
        if reason:
            detail_parts.append(f"reason={reason}")
        if error:
            detail_parts.append(f"error={error}")
        detail_parts.append(build_diagnostic_summary(diagnostics))

        return {
            "file": str(file_name),
            "status": str(status),
            "score": score,
            "score_display": format_accuracy_score(score),
            "method": str(method) if method else "-",
            "details": " | ".join([part for part in detail_parts if part]),
            "output": str(output_path) if output_path else "-",
        }

    def _show_conversion_report(self, rows, run_cancelled=False):
        if not rows:
            return

        success_count = sum(1 for row in rows if row.get("status") == "Success")
        failed_count = sum(1 for row in rows if row.get("status") == "Failed")
        skipped_count = sum(1 for row in rows if row.get("status") == "Skipped (impossible)")
        scored = [float(row["score"]) for row in rows if row.get("score") is not None]
        avg_score = (sum(scored) / len(scored)) if scored else None

        title = "Conversion Report"
        if run_cancelled:
            title += " (Partial)"

        report_window = Toplevel(self.root)
        report_window.title(title)
        report_window.geometry("1220x620")
        report_window.transient(self.root)
        report_window.grab_set()

        container = ttk.Frame(report_window, padding="10")
        container.grid(row=0, column=0, sticky="nsew")
        report_window.grid_rowconfigure(0, weight=1)
        report_window.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        summary = (
            f"Success: {success_count} | Failed: {failed_count} | "
            f"Skipped impossible: {skipped_count} | Total: {len(rows)}"
        )
        if avg_score is not None:
            summary += f" | Avg score: {avg_score:.1f} ({classify_accuracy_band(avg_score)})"
        if run_cancelled:
            summary += " | Run cancelled by user"

        summary_label = ttk.Label(container, text=summary)
        summary_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        table_frame = ttk.Frame(container)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        col_names = ("file", "status", "score", "method", "details", "output")
        tree = ttk.Treeview(table_frame, columns=col_names, show="headings")
        tree.grid(row=0, column=0, sticky="nsew")

        tree.heading("file", text="File")
        tree.heading("status", text="Status")
        tree.heading("score", text="Score")
        tree.heading("method", text="Method")
        tree.heading("details", text="Details")
        tree.heading("output", text="Output")

        tree.column("file", width=220, minwidth=160, stretch=True)
        tree.column("status", width=160, minwidth=130, stretch=False)
        tree.column("score", width=120, minwidth=90, stretch=False)
        tree.column("method", width=130, minwidth=100, stretch=False)
        tree.column("details", width=340, minwidth=220, stretch=True)
        tree.column("output", width=230, minwidth=180, stretch=True)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        for row in rows:
            tree.insert(
                "",
                "end",
                values=(
                    row.get("file", ""),
                    row.get("status", ""),
                    row.get("score_display", "-"),
                    row.get("method", "-"),
                    row.get("details", ""),
                    row.get("output", "-"),
                ),
            )

        button_frame = ttk.Frame(container)
        button_frame.grid(row=2, column=0, sticky="e", pady=(8, 0))
        ttk.Button(button_frame, text="Close", command=report_window.destroy).grid(row=0, column=0)

    def browse_images(self):
        """Open file dialog to select images."""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_paths:
            for file_path in file_paths:
                if not any(item.file_path == Path(file_path) for item in self.config.image_items):
                    img_item = ImageItem(file_path)
                    self.config.image_items.append(img_item)
                    item_id = str(id(img_item))
                    self.listbox.insert('', 'end', iid=item_id, 
                                       text=img_item.get_display_name(),
                                       values=(img_item.get_settings_summary(),))
            self.validate_inputs()
            
    def on_drop(self, event):
        """Handle drag and drop files."""
        files = self.root.tk.splitlist(event.data)
        
        for file in files:
            file_path = Path(file)
            if file_path.suffix.lower() in self.VALID_EXTENSIONS:
                if not any(item.file_path == file_path for item in self.config.image_items):
                    img_item = ImageItem(file_path)
                    self.config.image_items.append(img_item)
                    item_id = str(id(img_item))
                    self.listbox.insert('', 'end', iid=item_id,
                                       text=img_item.get_display_name(),
                                       values=(img_item.get_settings_summary(),))
                    
        self.validate_inputs()
        
    def clear_list(self):
        """Clear the file list."""
        for item in self.listbox.get_children():
            self.listbox.delete(item)
        self.config.image_items.clear()
        self.clear_preview()
        self.validate_inputs()
        
    def remove_selected(self):
        """Remove selected image from list."""
        selection = self.listbox.selection()
        if not selection:
            return
            
        item_id = selection[0]
        # Find and remove from image_items
        for img_item in self.config.image_items[:]:
            if str(id(img_item)) == item_id:
                self.config.image_items.remove(img_item)
                break
                
        # Remove from listbox
        self.listbox.delete(item_id)
        self.clear_preview()
        self.validate_inputs()
        
    def start_processing(self):
        """Start image processing in background thread."""
        if self.config.processing:
            return
            
        if not self.config.image_items:
            messagebox.showwarning("Warning", "No images selected. Please select some images to process.")
            return
        
        # Update UI state
        self.config.processing = True
        self.process_btn.state(['disabled'])
        self.cancel_btn.state(['!disabled'])
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.config.image_items)
        
        # Start processing thread
        thread = threading.Thread(
            target=self.process_images,
            daemon=True
        )
        thread.start()
        
    def process_images(self):
        """Process images in background thread."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        jobs = self._build_processing_jobs()
        archive_dir = self._resolve_impossible_archive_dir(timestamp)
        self.update_status("Analyzing sheets before processing...")

        processable_jobs, impossible_jobs, report_path = assess_and_archive_impossible_files(
            jobs,
            archive_folder=archive_dir,
            min_confidence=12.0,
        )

        total_files = len(processable_jobs)
        processed = 0
        failed = []
        run_cancelled = False
        conversion_rows = []

        try:
            self.root.after(
                0,
                lambda: self.progress.configure(value=0, maximum=max(1, total_files)),
            )

            for job in impossible_jobs:
                file_name = Path(job.get("image_path", "")).name
                diagnostics = job.get("diagnostics") or {}
                reason = str(job.get("reason", "impossible_file"))
                output_path = job.get("archived_path") or archive_dir
                conversion_rows.append(
                    self._build_conversion_row(
                        file_name=file_name,
                        status="Skipped (impossible)",
                        diagnostics=diagnostics,
                        reason=reason,
                        output_path=output_path,
                    )
                )

            if impossible_jobs:
                self.root.after(0, lambda: self._highlight_impossible_items(impossible_jobs))
                self.root.after(
                    0,
                    lambda: self._notify_impossible_jobs(impossible_jobs, archive_dir, report_path),
                )

            if total_files == 0:
                self.update_status("No processable images found. Check impossible files report.")
                if conversion_rows:
                    self.root.after(0, lambda rows=list(conversion_rows): self._show_conversion_report(rows))
                return

            for idx, job in enumerate(processable_jobs, 1):
                if not self.config.processing:
                    run_cancelled = True
                    break

                img_item = job["image_item"]
                settings = job["settings"]
                self.update_status(f"Processing {idx}/{total_files}: {img_item.file_path.name}")

                try:
                    output_folder = split_and_resize_image(
                        str(img_item.file_path),
                        settings["across"],
                        settings["high"],
                        settings["output_size"],
                        settings["folder"],
                        settings["format_setting"],
                        settings["smart_grid_setting"],
                        timestamp,
                        output_base_dir=settings["output_base_dir"],
                        crop_margin=settings["crop_margin"],
                        orientation_mode=settings["orientation_mode"],
                    )
                    processed += 1
                    diagnostics = self._load_conversion_diagnostics(output_folder, img_item.file_path.stem)
                    conversion_rows.append(
                        self._build_conversion_row(
                            file_name=img_item.file_path.name,
                            status="Success",
                            diagnostics=diagnostics,
                            output_path=output_folder,
                        )
                    )
                except Exception as img_exc:
                    failed.append((img_item.file_path.name, str(img_exc)))
                    conversion_rows.append(
                        self._build_conversion_row(
                            file_name=img_item.file_path.name,
                            status="Failed",
                            diagnostics={},
                            error=str(img_exc),
                        )
                    )

                self.root.after(0, lambda v=idx: self.progress.configure(value=v))

            if self.config.processing:
                if failed:
                    failed_preview = "\n".join([f"- {name}: {error}" for name, error in failed[:10]])
                    if len(failed) > 10:
                        failed_preview += f"\n... and {len(failed) - 10} more error(s)."
                    skipped = len(impossible_jobs)
                    self.update_status(
                        f"Completed with errors ({processed}/{total_files} successful, {skipped} skipped impossible)."
                    )
                    self._show_warning(
                        "Completed with errors",
                        (
                            f"Processed {processed}/{total_files} processable image(s).\n"
                            f"Skipped impossible: {len(impossible_jobs)}.\n\nErrors:\n{failed_preview}"
                        ),
                    )
                    if conversion_rows:
                        self.root.after(0, lambda rows=list(conversion_rows): self._show_conversion_report(rows))
                else:
                    skipped = len(impossible_jobs)
                    self.update_status(
                        f"Processing completed. Success: {processed}/{total_files}, skipped impossible: {skipped}."
                    )
                    self._show_info(
                        "Success",
                        (
                            f"Processed {processed}/{total_files} processable image(s) successfully.\n"
                            f"Skipped impossible files: {skipped}."
                        ),
                    )
                    if conversion_rows:
                        self.root.after(0, lambda rows=list(conversion_rows): self._show_conversion_report(rows))
            elif run_cancelled and conversion_rows:
                self.root.after(
                    0,
                    lambda rows=list(conversion_rows): self._show_conversion_report(rows, run_cancelled=True),
                )
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self._show_error("Error", f"An error occurred: {str(e)}")
        finally:
            self.config.processing = False
            self.root.after(0, self.reset_ui)

    def cancel_processing(self):
        """Cancel ongoing processing."""
        self.config.processing = False
        self.update_status("Processing cancelled")
        
    def update_status(self, message):
        """Update status message (thread-safe)."""
        self.root.after(0, lambda: self.status_var.set(message))

    def _show_info(self, title, message):
        self.root.after(0, lambda: messagebox.showinfo(title, message))

    def _show_warning(self, title, message):
        self.root.after(0, lambda: messagebox.showwarning(title, message))

    def _show_error(self, title, message):
        self.root.after(0, lambda: messagebox.showerror(title, message))
        
    def reset_ui(self):
        """Reset UI to ready state."""
        self.process_btn.state(['!disabled'])
        self.cancel_btn.state(['disabled'])
        self.validate_inputs()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Image Splitter and Resizer")
    parser.add_argument('files', nargs='*', help="List of image files to process")
    parser.add_argument('--across', type=int, default=1, help="Number of images across (default: 1)")
    parser.add_argument('--high', type=int, default=1, help="Number of images high (default: 1)")
    parser.add_argument(
        '--quality',
        choices=['max', 'balanced', 'compact'],
        default='max',
        help="Output quality mode: max keeps native crop size, balanced=2048, compact=1024",
    )
    parser.add_argument(
        '--orientation',
        choices=['auto', 'horizontal', 'vertical'],
        default='auto',
        help="Force crop orientation: auto, horizontal, or vertical",
    )
    parser.add_argument('--crop-margin', type=int, default=0, help="Extra crop margin in pixels around each detected cell (default: 0)")
    parser.add_argument('--folder', type=str, help="Custom output folder name (optional)")
    parser.add_argument('--output-dir', type=str, help="Base output directory (optional)")
    args = parser.parse_args()

    if args.files:
        # CLI mode
        file_paths = args.files
        custom_folder = args.folder
        images_across = args.across
        images_high = args.high
        quality_to_size = {
            "max": None,
            "balanced": 2048,
            "compact": 1024,
        }
        output_size = quality_to_size[args.quality]
        maintain_format = False
        smart_grid = True
        crop_margin = args.crop_margin
        orientation_mode = args.orientation
        output_base_dir = args.output_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Processing {len(file_paths)} image(s)...")
        for idx, file_path in enumerate(file_paths, 1):
            print(f"  [{idx}/{len(file_paths)}] {Path(file_path).name}")
            split_and_resize_image(
                file_path,
                images_across,
                images_high,
                output_size,
                custom_folder,
                maintain_format,
                smart_grid,
                timestamp,
                output_base_dir=output_base_dir,
                crop_margin=crop_margin,
                orientation_mode=orientation_mode,
            )
        print("Processing completed!")
    else:
        # GUI mode
        root = TkinterDnD.Tk()
        app = ImageSplitterGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()

