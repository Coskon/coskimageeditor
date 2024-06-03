import copy
import io
import time
import tkinter as tk
import image as im
import numpy as np
import ast
import os
import traceback
import re
import imageio as iio
import requests
import pyautogui
import win32clipboard
from io import BytesIO
from psd_tools import PSDImage
from tkinter import filedialog, simpledialog, messagebox, ttk, colorchooser
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk, ImageGrab
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

VIDEO_SUPPORTED = "*.mp4;*.avi;*.mkv;*.mov;*.mpeg;*.webm"
FILE_SUPPORTED = "*.gif;*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.jfif;*.tif;*.ppm;*.pgm;*.pnm;*.webp;*.ico;*.psd;*.cur"
SAVE_SUPPORTED = "*.gif;*.png;*.jpg;*.jpeg;*.jfif;*.bmp;*.tiff;*.tif;*.ppm;*.pgm;*.pnm;*.webp;*.ico"
FILETYPES = FILE_SUPPORTED.replace("*", "").split(";")
SAVE_FILETYPES = SAVE_SUPPORTED.replace("*", "").split(";")
VIDEO_FILETYPES = VIDEO_SUPPORTED.replace("*", "").split(";")
Image.MAX_IMAGE_PIXELS = None
COMMAND_LIST = [
    '-help', '-exit', '-quit', '-close', '-optimize', '-filter', '-kernel', '-load', '-save', '-clc', '-clear',
    '-current', '-resize', '-fps', '-undo', '-redo', '-speed'
]
COMMAND_LIST_DISPLAY = [
    '-help', '-exit or -quit or -close', '-optimize', '-filter', '-kernel', '-load', '-save', '-clc', '-clear',
    '-current', '-resize', '-fps', '-undo and -redo', '-speed'
]
USAGE_LIST = [
    '  // Show list of commands.', '  // Close the console', '  // Enable or disable optimization techniques',
    ', --filter_type --filter_arg1 value1 ...  // Apply filter', ', --kernel_type --kernel_arg1 value1 ...  // Apply kernel',
    ', --file_path  // Load image from file', ', --save_path  // Save image to file', '  // Clear the console',
    '  // Empties canvas | Removes user defined variables', '  // Show current image loaded',
    ', --w --h --mantain_aspect_ratio --method  // Resize image',
    '  // Show FPS', ', --set_limit (optional)  // Go back to the previous/next actions or change the limit',
    ' --speed --type  // Set speed (type: absolute, relative)'
]


def convert_to_type(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None


def check_type(s):
    if '[' in s and ']' in s:
        s = s.replace('[','').replace(']','').split(',')
        return [convert_to_type(i) for i in s]
    return convert_to_type(s)

def is_gif(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == ".gif"


def is_image(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in FILETYPES


def is_video(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in VIDEO_FILETYPES


def get_image_size(file_path):
    try:
        if file_path.endswith('.psd'):
            width, height = PSDImage.open(file_path).size
            return width, height
        else:
            with Image.open(file_path) as img:
                width, height = img.size
                return width, height
    except Image.DecompressionBombError as e:
        print(f"DecompressionBombError: {e}")
        Image.MAX_IMAGE_PIXELS = 178956970
        return None
    except Exception as e:
        print(f"Error: {e}")
        Image.MAX_IMAGE_PIXELS = 178956970
        return None


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        try:
            float(s)
            return True
        except ValueError:
            return False


def coma_split(text):
    text_list = []
    stop, prt = False, False
    curr_text = ""
    for char in text:
        if char in '({[': prt = True
        if char == ',' and not prt: stop = True
        if stop:
            text_list.append(curr_text)
            curr_text = ""
            stop = False
        else:
            curr_text += char
        if char in ')}]': prt = False
    text_list.append(curr_text)
    return text_list


def ImageEditor(fullscreen=False, theme="dark", open_console=False, transparentbg=True, unit="px", DPI=96, action_size=32):
    root = TkinterDnD.Tk()
    root.title("Cosk Image Editor")
    root.attributes('-fullscreen', fullscreen)
    ImageEditorWindow(root, theme, open_console, transparentbg, unit, DPI, action_size)
    root.mainloop()


class ResizeWindow(tk.simpledialog.Dialog):
    def __init__(self, parent, orig_size, theme):
        self.orig_size = orig_size
        self.scale = 1
        self.width = None
        self.height = None

        if theme == "dark":
            self.bg_color = "#444444"
            self.button_color = "#343434"
            self.text_color = "#FFFFFF"
        else:
            self.bg_color = "#ABABAB"
            self.button_color = "#EEEEEE"
            self.text_color = "black"

        self.resize_method = tk.StringVar(value="Nearest")
        super().__init__(parent)

    def body(self, master):
        self.configure(bg=self.bg_color)
        master.configure(bg=self.bg_color)
        self.resizable(False, False)

        self.width_label = tk.Label(master, text="Width:", bg=self.bg_color, fg=self.text_color)
        self.width_entry = tk.Entry(master)
        self.width_label.grid(row=0, column=0, padx=5, pady=5)
        self.width_entry.grid(row=0, column=1, padx=5, pady=5)

        self.height_label = tk.Label(master, text="Height:", bg=self.bg_color, fg=self.text_color)
        self.height_entry = tk.Entry(master)
        self.height_label.grid(row=1, column=0, padx=5, pady=5)
        self.height_entry.grid(row=1, column=1, padx=5, pady=5)


        self.maintain_aspect_ratio = tk.BooleanVar(value=True)
        self.aspect_ratio_checkbox = tk.Checkbutton(
            master, text="Maintain Aspect Ratio", variable=self.maintain_aspect_ratio, bg=self.bg_color, fg=self.text_color,
                                                                  selectcolor=self.button_color, activebackground=self.bg_color,
                                                                  activeforeground=self.text_color
        )
        self.aspect_ratio_checkbox.grid(row=2, column=0, columnspan=2, pady=5)

        self.method_label = tk.Label(master, text="Resizing method:", bg=self.bg_color, fg=self.text_color)
        self.method_combobox = ttk.Combobox(master, textvariable=self.resize_method,
                                            values=["Nearest", "Bilinear", "Bicubic"])
        self.method_label.grid(row=3, column=0, padx=5, pady=5)
        self.method_combobox.grid(row=3, column=1, padx=5, pady=5)

        self.maintain_aspect_ratio.trace_add("write", self.on_aspect_ratio_change)
        self.width_entry.bind("<KeyRelease>", self.on_aspect_ratio_change)

        return self.width_entry

    def on_aspect_ratio_change(self, *args):
        width_content = self.width_entry.get()
        if self.maintain_aspect_ratio.get():
            self.height_entry.configure(state="disabled")
            if not width_content.isdigit(): return None
            self.height_entry.configure(state="normal")
            new_width = int(width_content)
            aspect_ratio = self.orig_size[1] / self.orig_size[0]
            new_height = int(new_width * aspect_ratio)
            self.height_entry.delete(0, tk.END)
            self.height_entry.insert(0, str(new_height))
            self.height_entry.configure(state="disabled")
        else:
            self.height_entry.delete(0, tk.END)
            self.height_entry.configure(state="normal")

    def apply(self):
        if not self.width_entry.get().isdigit() or not self.height_entry.get().isdigit():
            self.width, self.height = None, None
            return None
        self.width = int(self.width_entry.get()) if self.width_entry.get() else None
        self.height = int(self.height_entry.get()) if self.height_entry.get() else None
        self.resize_method = self.resize_method.get()


class NoiseWindow(tk.simpledialog.Dialog):
    def __init__(self, parent, size, theme):
        if theme == "dark":
            self.bg_color = "#444444"
            self.button_color = "#343434"
            self.text_color = "#FFFFFF"
        else:
            self.bg_color = "#ABABAB"
            self.button_color = "#EEEEEE"
            self.text_color = "black"

        self.strength = None
        self.transparent_check = False
        self.invert_check = False
        self.imsize = size
        self.noise_type = "random"
        self.noise_type_stringvar = tk.StringVar(value="Random")
        super().__init__(parent)

    def body(self, master):
        self.configure(bg=self.bg_color)
        master.configure(bg=self.bg_color)
        self.resizable(False, False)

        self.strength_label = tk.Label(master, text="Strength:", bg=self.bg_color, fg=self.text_color)
        self.strength_entry = tk.Entry(master, textvariable=self.strength)
        self.strength_label.grid(row=0, column=0, padx=5, pady=5)
        self.strength_entry.grid(row=0, column=1, padx=5, pady=5)

        self.transparent_check = tk.BooleanVar(value=False)
        self.transparent_checkbox = tk.Checkbutton(
            master, text="Transparent", variable=self.transparent_check, bg=self.bg_color, fg=self.text_color,
                                                                  selectcolor=self.button_color, activebackground=self.bg_color,
                                                                  activeforeground=self.text_color
        )
        self.transparent_checkbox.grid(row=2, column=0, columnspan=1, pady=5)

        self.invert_check = tk.BooleanVar(value=False)
        self.invert_checkbox = tk.Checkbutton(
            master, text="Invert", variable=self.invert_check, bg=self.bg_color, fg=self.text_color,
            selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.invert_checkbox.grid(row=2, column=0, columnspan=2, pady=5)

        self.type_label = tk.Label(master, text="Noise Type:", bg=self.bg_color, fg=self.text_color)
        self.type_combobox = ttk.Combobox(master, textvariable=self.noise_type_stringvar,
                                            values=["Random", "ColorRandom", "Perlin", "Rain", "Circles", "Serpent"])
        self.grid = self.type_label.grid(row=3, column=0, padx=5, pady=5)
        self.type_combobox.grid(row=3, column=1, padx=5, pady=5)

        return self.strength_entry

    def apply(self):
        if not is_number(self.strength_entry.get()):
            self.strength = None
            return None
        self.strength = float(self.strength_entry.get()) if self.strength_entry.get() else None
        self.noise_type = self.noise_type_stringvar.get()
        self.transparent_check = self.transparent_check.get()
        self.invert_check = self.invert_check.get()


class FilterWindow(tk.simpledialog.Dialog):
    def __init__(self, parent, theme, images):
        if theme == "dark":
            self.bg_color = "#444444"
            self.button_color = "#343434"
            self.text_color = "#FFFFFF"
        else:
            self.bg_color = "#ABABAB"
            self.button_color = "#EEEEEE"
            self.text_color = "black"

        self.textboxes = [0, 0, 0, 0, 0]
        self.options_entries = []
        self.changed_palette = False
        self.images_preview = images.copy()

        self.pixelization = 1
        self.filter_type = "identity"
        self.custom_filter = [1, 1, 1, 1, 0]
        self.palette = ""
        self.palette_selected = "Sepia"
        self.dither_type = ""
        self.dither_colors = 4
        self.saturation = 1
        self.contrast = 1
        self.luminance = 1
        self.colors = 16
        self.mirror = None
        self.greyscale = False
        self.one_palette = True
        self.pre_palette = None
        self.frame_num = 0
        self.bayer_strength = 16
        self.bayer_size = 2

        super().__init__(parent)

    def body(self, master):
        self.configure(bg=self.bg_color)
        master.configure(bg=self.bg_color)
        self.resizable(False, False)

        filter_types = list(im.FILTERS.keys())
        for i in range(len(filter_types)):
            if filter_types[i] == "identity":
                filter_types[i] = "No Filter"
            elif filter_types[i] == "changepalette":
                filter_types[i] = "Change Palette"
            else:
                filter_types[i] = filter_types[i].capitalize()
        filter_types.remove("Identity_tuple")
        dither_types = ["Floyd-Steinberg", "Atkinson", "Stucki", "Halftone"]
        mirror_types = ["Vertical", "Horizontal"]

        # FILTER TYPE
        self.type_label = tk.Label(master, text="Filter Type:", bg=self.bg_color, fg=self.text_color)
        self.type_combobox = ttk.Combobox(master, textvariable=self.filter_type,
                                          values=filter_types)
        self.type_combobox.set("No Filter")
        self.grid = self.type_label.grid(row=0, column=0, padx=5, pady=5)
        self.type_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.type_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_values(id=1))

        self.custom_frame = tk.Frame(master, bg=self.bg_color)
        self.palette_frame = tk.Frame(master, bg=self.bg_color)
        self.custom_frame.grid(row=2, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.palette_frame.grid(row=2, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.custom_frame.grid_remove()
        self.palette_frame.grid_remove()

        self.sat_label = tk.Label(master, text=f"Saturation ({self.saturation:.2f})", bg=self.bg_color, fg=self.text_color)
        self.con_label = tk.Label(master, text=f"Contrast ({self.contrast:.2f})", bg=self.bg_color, fg=self.text_color)
        self.lum_label = tk.Label(master, text=f"Luminance ({self.luminance:.2f})", bg=self.bg_color, fg=self.text_color)
        self.greycol_label = tk.Label(master, text=f"Greyscale Colors", bg=self.bg_color,
                                  fg=self.text_color)
        self.sat_entry = tk.Scale(master, from_=1, to=200, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.con_entry = tk.Scale(master, from_=1, to=200, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.lum_entry = tk.Scale(master, from_=1, to=200, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.greycol_entry = tk.Entry(master, textvariable=self.colors)
        self.sat_entry.set(100)
        self.con_entry.set(100)
        self.lum_entry.set(100)
        self.sat_label.grid(row=3, column=0, padx=5, pady=5)
        self.con_label.grid(row=3, column=1, padx=5, pady=5)
        self.lum_label.grid(row=3, column=2, padx=5, pady=5)
        self.greycol_label.grid(row=3, column=3, padx=5, pady=5)
        self.sat_entry.grid(row=4, column=0, padx=5, pady=5)
        self.con_entry.grid(row=4, column=1, padx=5, pady=5)
        self.lum_entry.grid(row=4, column=2, padx=5, pady=5)
        self.greycol_entry.grid(row=4, column=3, padx=5, pady=5)

        # PIXELIZATION
        self.pixellate = tk.BooleanVar(value=False)
        self.pixellate_checkbox = tk.Checkbutton(
            master, text="Pixelate", variable=self.pixellate, command=lambda: self.update_values(id=1),
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.pixellate_checkbox.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.pix_label = tk.Label(master, text="Pixelization:", bg=self.bg_color, fg=self.text_color)
        self.pix_entry = tk.Entry(master, textvariable=self.pixelization)
        self.pix_label.grid(row=5, column=1, padx=5, pady=5)
        self.pix_entry.grid(row=5, column=2, padx=5, pady=5)
        self.options_entries.append([self.pixellate, self.pix_label, self.pix_entry])

        # DITHERING
        self.dithering = tk.BooleanVar(value=False)
        self.dithering_checkbox = tk.Checkbutton(
            master, text="Dithering", variable=self.dithering, command=lambda: self.update_values(id=1),
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.dithering_checkbox.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.dither_label = tk.Label(master, text="Dither Type/N° Colors:", bg=self.bg_color, fg=self.text_color)
        self.dither_combobox = ttk.Combobox(master, textvariable=self.dither_type, values=dither_types)
        self.dither_combobox.set("Floyd-Steinberg")
        self.dither_combobox.bind("<<ComboboxSelected>>", self.update_values)
        self.dither_col_entry = tk.Entry(master, textvariable=self.dither_colors, width=10)
        self.dither_label.grid(row=6, column=1, padx=5, pady=5)
        self.dither_combobox.grid(row=6, column=2, padx=5, pady=5)
        self.dither_col_entry.grid(row=6, column=3, padx=5, pady=5)
        self.options_entries.append([self.dithering, self.dither_label, self.dither_col_entry, self.dither_combobox])

        # MIRROR
        self.mirroring = tk.BooleanVar(value=False)
        self.mirroring_checkbox = tk.Checkbutton(
            master, text="Mirroring", variable=self.mirroring, command=lambda: self.update_values(id=1),
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.mirroring_checkbox.grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.mirror_label = tk.Label(master, text="Mirror Type:", bg=self.bg_color, fg=self.text_color)
        self.mirror_combobox = ttk.Combobox(master, textvariable=self.mirror, values=mirror_types)
        self.mirror_combobox.set("Vertical")
        self.mirror_label.grid(row=7, column=1, padx=5, pady=5)
        self.mirror_combobox.grid(row=7, column=2, padx=5, pady=5)
        self.mirror_combobox.bind("<<ComboboxSelected>>", self.update_values)
        self.options_entries.append([self.mirroring, self.mirror_label, self.mirror_combobox])

        # ONE PALETTE
        self.use_one_palette = tk.BooleanVar(value=self.one_palette)
        self.one_palette_check = tk.Checkbutton(
            master, text="Use one palette for all frames", variable=self.use_one_palette, command=self.update_values,
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.one_palette_check.grid(row=8, column=0, padx=5, pady=5, sticky="w")

        # BAYERS SETTINGS
        self.bayer_size_label = tk.Label(master, text=f"Bayer Strength ({self.bayer_strength:.2f})", bg=self.bg_color,
                                  fg=self.text_color)
        self.bayer_strength_label = tk.Label(master, text=f"Bayer Matrix Size ({self.bayer_size})", bg=self.bg_color, fg=self.text_color)
        self.bayer_size_entry = tk.Scale(master, from_=1, to=5, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.bayer_strength_entry = tk.Scale(master, from_=1, to=50, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.bayer_size_entry.set(self.bayer_size)
        self.bayer_strength_entry.set(self.bayer_strength)
        self.bayer_size_label.grid(row=9, column=0, padx=5, pady=5)
        self.bayer_strength_label.grid(row=9, column=1, padx=5, pady=5)
        self.bayer_size_entry.grid(row=10, column=0, padx=5, pady=5)
        self.bayer_strength_entry.grid(row=10, column=1, padx=5, pady=5)

        for i in range(len(self.images_preview)):
            w, h = self.images_preview[i].size
            h2 = 400
            w2 = int(w*(h2/h))
            self.images_preview[i] = self.images_preview[i].resize((w2, h2))
        self.photoimg = ImageTk.PhotoImage(self.images_preview[self.frame_num])
        self.pre_palette = im.findPalette(np.array(self.images_preview[self.frame_num]),
                                          num_colors=self.dither_colors)
        self.image_label = tk.Label(master, image=self.photoimg, highlightthickness=3, highlightbackground=self.button_color,
                                    bg=self.bg_color, fg=self.text_color)
        self.image_text = tk.Label(master, text="Rough Preview", bg=self.bg_color, fg=self.text_color)
        self.frame_text = tk.Label(master, text=f"Current Frame ({self.frame_num})", bg=self.bg_color, fg=self.text_color)
        self.frame_scale = tk.Scale(master, from_=0, to=len(self.images_preview)-1, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.image_label.grid(row=0,rowspan=7,column=5,padx=5,pady=5)
        self.image_text.grid(row=7, column=5, padx=5, pady=5)
        self.frame_text.grid(row=8, column=5, padx=5, pady=5)
        self.frame_scale.grid(row=9, column=5, padx=5, pady=5)

        self.pix_entry.bind("<KeyRelease>", self.update_values)
        self.greycol_entry.bind("<KeyRelease>", self.update_values)
        self.dither_col_entry.bind("<KeyRelease>", self.update_values)

        self.update_widgets()

    def update_values(self, event=None, id=None):
        if id == 1: self.update_widgets()
        self.saturation = self.sat_entry.get() / 100
        self.contrast = self.con_entry.get() / 100
        self.luminance = self.lum_entry.get() / 100
        self.sat_label.config(text=f"Saturation ({self.saturation:.2f})")
        self.con_label.config(text=f"Contrast ({self.contrast:.2f})")
        self.lum_label.config(text=f"Luminance ({self.luminance:.2f})")
        self.frame_num = int(self.frame_scale.get())
        self.frame_text.configure(text=f"Current Frame ({self.frame_num})")

        self.pixelization = int(self.pix_entry.get()) if is_number(self.pix_entry.get()) and self.pixellate.get() else 1

        self.filter_type = self.type_combobox.get() if self.type_combobox.get() else "identity"
        if self.filter_type == "Change Palette":
            self.filter_type = "changepalette"
        elif self.filter_type == "No Filter":
            self.filter_type = "identity"

        self.one_palette = bool(self.use_one_palette.get())
        self.bayer_size = int(self.bayer_size_entry.get())
        self.bayer_strength = self.bayer_strength_entry.get()
        self.bayer_strength_label.config(text=f"Bayer Strength ({self.bayer_strength:.2f})")
        self.bayer_size_label.config(text=f"Bayer Matrix Size ({self.bayer_size})")

        self.dither_type = self.dither_combobox.get() if self.dithering.get() else None
        tmp_col = self.dither_colors
        self.dither_colors = int(self.dither_col_entry.get()) if self.dithering.get() and self.dither_col_entry.get() != ""  else self.dither_colors
        self.colors = int(self.greycol_entry.get()) if self.greycol_entry.get() and self.greyscale else self.colors
        self.mirror = self.mirror_combobox.get() if self.mirroring.get() and self.mirroring.get() else None
        if tmp_col != self.dither_colors or not self.one_palette:
            if self.filter_type == "Greyscale":
                self.pre_palette = (np.tile(np.arange(0, 255 + 255 / self.colors, 255 / self.colors), (4, 1)).T).astype(np.uint8)
                self.pre_palette[:, 3] = 255
            else:
                self.pre_palette = im.findPalette(np.array(self.images_preview[self.frame_num]), num_colors=self.dither_colors)
        self.palette = self.palette_combobox.get().lower() if self.changed_palette else self.pre_palette

        try:
            self.custom_filter = [float(i.get()) for i in self.textboxes] if self.textboxes else [1, 1, 1, 1, 0]
            if any([val in ['', None] for val in self.custom_filter]): self.custom_filter = [1, 1, 1, 1, 0]
        except:
            None

        img = im.imageFilter(np.array(self.images_preview[self.frame_num]), self.filter_type, customFilter=self.custom_filter, luminance=self.luminance,
                             contrast=self.contrast, saturation=self.saturation, colors=self.colors,
                             ditherType=self.dither_type, ditherColors=self.dither_colors, pixelSize=self.pixelization,
                             customPalette=self.palette, mirror=self.mirror, bayer=(self.bayer_size, self.bayer_strength))
        self.photoimg = ImageTk.PhotoImage(Image.fromarray(img))
        self.image_label.config(image=self.photoimg)

    def update_widgets(self, event=None):
        selected_type = self.type_combobox.get()

        # EXTRA PARAMETERS
        for options in self.options_entries:
            a = options[0] if isinstance(options[0], bool) else options[0].get()
            if a:
                for i in options[1:]: i.config(state="normal")
            else:
                for i in options[1:]: i.config(state="disabled")

        # CHANGED PALETTE / CUSTOM FILTER
        if self.changed_palette:
            self.palette_selected = self.palette_combobox.get()
            self.palette_combobox.destroy()
        self.palette_label.destroy() if hasattr(self, 'palette_label') else None
        self.changed_palette = False
        self.greycol_label.config(state="disabled")
        self.greycol_entry.config(state="disabled")
        self.greyscale = False
        if selected_type == "Change Palette":
            self.changed_palette = True

            palettes = list(im.PALETTES.keys())
            palettes = [i.capitalize() for i in palettes]
            self.palette_label = tk.Label(self.palette_frame, text="Palette:", bg=self.bg_color, fg=self.text_color)
            self.palette_label.pack(side=tk.LEFT, padx=5)
            self.palette_combobox = ttk.Combobox(self.palette_frame, textvariable=self.palette, values=palettes)
            self.palette_combobox.set(self.palette_selected)
            self.palette_combobox.bind("<<ComboboxSelected>>", self.update_values)
            self.palette_combobox.pack()
            self.palette_frame.grid()

            self.custom_frame.grid_remove()
        elif selected_type == "Custom":
            for widget in self.custom_frame.winfo_children():
                widget.destroy()

            custom_text = tk.Label(self.custom_frame, text="(RGBAI)", bg=self.bg_color, fg=self.text_color)
            custom_text.pack(side=tk.LEFT, padx=5)

            for i in range(5):
                textbox = tk.Entry(self.custom_frame, width=5)
                textbox.pack(side=tk.RIGHT, padx=5)
                textbox.bind("<KeyRelease>", self.update_values)
                self.textboxes[i] = textbox
            self.textboxes.reverse()
            self.custom_frame.grid()

            self.palette_combobox.destroy() if hasattr(self, 'palette_combobox') else None
            self.palette_frame.grid_remove()
        elif selected_type == "Greyscale":
            self.greyscale = True
            self.greycol_label.config(state="normal")
            self.greycol_entry.config(state="normal")
            self.custom_frame.grid_remove()
            self.palette_combobox.destroy() if hasattr(self, 'palette_combobox') else None
            self.palette_frame.grid_remove()
        else:
            self.custom_frame.grid_remove()
            self.palette_combobox.destroy() if hasattr(self, 'palette_combobox') else None
            self.palette_frame.grid_remove()

    def apply(self):
        if not is_number(self.pix_entry.get()):
            self.pixelization = None
        self.pixelization = int(self.pix_entry.get()) if self.pix_entry.get() and self.pixellate.get() else 1

        self.filter_type = self.type_combobox.get() if self.type_combobox.get() else "identity"
        if self.filter_type == "Change Palette":
            self.filter_type = "changepalette"
        elif self.filter_type == "No Filter":
            self.filter_type = "identity"

        self.palette = self.palette_combobox.get() if self.changed_palette else None

        self.dither_type = self.dither_combobox.get() if self.dithering.get() and self.dithering.get() else None
        self.dither_colors = self.dither_col_entry.get() if self.dithering.get() and self.dithering.get() else self.dither_colors
        self.colors = int(self.greycol_entry.get()) if self.greycol_entry.get() and self.greyscale else self.colors
        self.mirror = self.mirror_combobox.get() if self.mirroring.get() and self.mirroring.get() else None

        try: self.custom_filter = [float(i.get()) for i in self.textboxes] if self.textboxes else [1, 1, 1, 1, 0]
        except: return


class AsciiWindow(tk.simpledialog.Dialog):
    def __init__(self, parent, theme, images):
        if theme == "dark":
            self.bg_color = "#444444"
            self.button_color = "#343434"
            self.text_color = "#FFFFFF"
        else:
            self.bg_color = "#ABABAB"
            self.button_color = "#EEEEEE"
            self.text_color = "black"

        self.images = images.copy()
        self.size = 100
        self.ascii_type = "Braille"
        self.contrast = 1
        self.invert = True
        self.dither_type = "No Dithering"
        self.smooth = False
        self.accentuate = False
        self.accentuate_thickness = 1
        self.font_size = 5
        self.braille_space = False
        self.space_strip = True
        self.curr_frame = 0
        self.threshold_lower = 0
        self.threshold_upper = 255
        #self.unihan_chars = im.unihanCharacters()

        super().__init__(parent)

    def body(self, master):
        self.configure(bg=self.bg_color)
        master.configure(bg=self.bg_color)
        self.resizable(False, False)

        ascii_types = ["Braille", "Braille6", "Squares", "Simple", "Complex"]#, "Unihan"]
        dither_types = ["No Dithering", "Floyd-Steinberg", "Atkinson", "Stucki"]

        # ASCII TYPE
        self.type_label = tk.Label(master, text="Type:", bg=self.bg_color, fg=self.text_color)
        self.type_combobox = ttk.Combobox(master, textvariable=self.ascii_type,
                                          values=ascii_types)
        self.type_combobox.set("Braille")
        self.type_label.grid(row=0, column=0, padx=5, pady=5)
        self.type_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.type_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_values())

        # SIZE AND CONTRAST
        self.size_label = tk.Label(master, text=f"Size ({self.size})", bg=self.bg_color, fg=self.text_color)
        self.con_label = tk.Label(master, text=f"Contrast ({self.contrast:.2f})", bg=self.bg_color, fg=self.text_color)
        self.thr1_label = tk.Label(master, text=f"Threshold Lower ({self.threshold_lower})", bg=self.bg_color, fg=self.text_color)
        self.thr2_label = tk.Label(master, text=f"Threshold Upper ({self.threshold_upper})", bg=self.bg_color, fg=self.text_color)
        self.size_entry = tk.Scale(master, from_=10, to=500, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.con_entry = tk.Scale(master, from_=1, to=200, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.thr1_entry = tk.Scale(master, from_=1, to=255, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.thr2_entry = tk.Scale(master, from_=1, to=255, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.size_entry.set(self.size)
        self.con_entry.set(int(self.contrast*100))
        self.thr1_entry.set(self.threshold_lower)
        self.thr2_entry.set(self.threshold_upper)
        self.size_label.grid(row=1, column=0, padx=5, pady=5)
        self.con_label.grid(row=2, column=0, padx=5, pady=5)
        self.thr1_label.grid(row=1, column=2, padx=5, pady=5)
        self.thr2_label.grid(row=2, column=2, padx=5, pady=5)
        self.size_entry.grid(row=1, column=1, padx=5, pady=5)
        self.con_entry.grid(row=2, column=1, padx=5, pady=5)
        self.thr1_entry.grid(row=1, column=3, padx=5, pady=5)
        self.thr2_entry.grid(row=2, column=3, padx=5, pady=5)

        # DITHERING
        self.dither_label = tk.Label(master, text="Dithering:", bg=self.bg_color, fg=self.text_color)
        self.dithering_combobox = ttk.Combobox(master, textvariable=self.dither_type,
                                          values=dither_types)
        self.dithering_combobox.set("No Dithering")
        self.dither_label.grid(row=3, column=0, padx=5, pady=5)
        self.dithering_combobox.grid(row=3, column=1, padx=5, pady=5)
        self.dithering_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_values())

        # INVERT
        self.inverting = tk.BooleanVar(value=self.invert)
        self.inverting_checkbox = tk.Checkbutton(
            master, text="Invert", variable=self.inverting, command=self.update_values,
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.inverting_checkbox.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        # SMOOTH
        self.smoothing = tk.BooleanVar(value=self.smooth)
        self.smoothing_checkbox = tk.Checkbutton(
            master, text="Smooth", variable=self.smoothing, command=self.update_values,
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.smoothing_checkbox.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # ACCENTUATE
        self.accentuating = tk.BooleanVar(value=self.accentuate)
        self.accentuating_checkbox = tk.Checkbutton(
            master, text="Accentuate", variable=self.accentuating, command=self.update_values,
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.accentuating_checkbox.grid(row=5, column=0, padx=5, pady=5, sticky="w")

        # ACCENTUATE THICKNESS
        self.acc_label = tk.Label(master, text=f"Accentuate Thickness ({self.accentuate_thickness})", bg=self.bg_color, fg=self.text_color)
        self.acc_entry = tk.Scale(master, from_=0, to=3, orient="horizontal", showvalue=False,
                                   bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                   troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.acc_entry.set(self.accentuate_thickness)
        self.acc_label.grid(row=5, column=1, padx=5, pady=5)
        self.acc_entry.grid(row=5, column=2, padx=5, pady=5)

        # FONT SIZE
        self.font_label = tk.Label(master, text=f"Font Size ({self.font_size})", bg=self.bg_color,
                                  fg=self.text_color)
        self.font_entry = tk.Scale(master, from_=4, to=16, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.font_entry.set(self.font_size)
        self.font_label.grid(row=6, column=0, padx=5, pady=5)
        self.font_entry.grid(row=6, column=1, padx=5, pady=5)

        # BRAILE-SPACE, SPACE STRIP
        self.braille_empty = tk.BooleanVar(value=self.braille_space)
        self.strip_space = tk.BooleanVar(value=self.space_strip)
        self.braille_empty_check = tk.Checkbutton(
            master, text="Replace Empty", variable=self.braille_empty, command=self.update_values,
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.strip_space_check = tk.Checkbutton(
            master, text="Strip End Spaces", variable=self.strip_space, command=self.update_values,
            bg=self.bg_color, fg=self.text_color, selectcolor=self.button_color, activebackground=self.bg_color,
            activeforeground=self.text_color
        )
        self.braille_empty_check.grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.strip_space_check.grid(row=7, column=1, padx=5, pady=5, sticky="w")

        # CURRENT FRAME
        self.frame_label = tk.Label(master, text=f"Current Frame ({self.curr_frame+1})", bg=self.bg_color,
                                  fg=self.text_color)
        self.choose_frame = tk.Scale(master, from_=0, to=len(self.images)-1, orient="horizontal", showvalue=False,
                                  bg=self.bg_color, fg=self.text_color, activebackground=self.bg_color,
                                  troughcolor=self.button_color, highlightthickness=0, command=self.update_values)
        self.choose_frame.set(self.curr_frame)
        self.frame_label.grid(row=8, column=0, padx=5, pady=5)
        self.choose_frame.grid(row=8, column=1, padx=5, pady=5)

        self.copy_button = tk.Button(master, text="Copy", command=self.copy_to_clipboard)
        self.copy_button.grid(row=10, column=0, padx=5, pady=5)

        self.textframe = tk.Frame(master, bg=self.bg_color)
        self.textframe.grid(row=0, column=4, rowspan=11, padx=5, pady=5, sticky=tk.E)

        self.txtbox = tk.Text(self.textframe, bg=self.bg_color, fg=self.text_color, font=("TkFixedFont", self.font_size))
        self.txtbox.pack(fill=tk.BOTH, expand=True)

        self.txt_count = tk.Label(self.textframe, bg=self.bg_color, fg=self.text_color)
        self.txt_count.pack(side=tk.BOTTOM)

        self.textframe.grid_rowconfigure(0, weight=1)
        self.textframe.grid_columnconfigure(0, weight=1)

    def update_values(self, event=None):
        self.txtbox.delete("1.0", tk.END)
        self.size = int(self.size_entry.get())
        self.contrast = float(self.con_entry.get())/100
        self.threshold_lower = int(self.thr1_entry.get())
        self.threshold_upper = int(self.thr2_entry.get())
        self.ascii_type = self.type_combobox.get()
        self.invert = bool(self.inverting.get())
        self.dither_type = self.dithering_combobox.get()
        self.dither_type = None if self.dither_type == "No Dithering" else self.dither_type.lower()
        self.smooth = bool(self.smoothing.get())
        self.accentuate = bool(self.accentuating.get())
        self.accentuate_thickness = int(self.acc_entry.get())
        self.font_size = int(self.font_entry.get())
        self.braille_space = bool(self.braille_empty.get())
        self.space_strip = bool(self.strip_space.get())
        self.size_label.configure(text=f"Size ({self.size})")
        self.con_label.configure(text=f"Contrast ({self.contrast:.2f})")
        self.thr1_label.configure(text=f"Threshold Lower ({self.threshold_lower})")
        self.thr2_label.configure(text=f"Threshold Upper ({self.threshold_upper})")
        self.acc_label.configure(text=f"Accentuate Thickness ({self.accentuate_thickness})")
        self.curr_frame = int(self.choose_frame.get())
        self.frame_label.configure(text=f"Current Frame ({self.curr_frame+1})")
        ascii_img, img = im.imageToAscii(np.array(self.images[self.curr_frame].convert("RGBA")), self.size, asciiString=self.ascii_type,
                                    contrast=self.contrast, invert=self.invert, dither_type=self.dither_type,
                                    smooth=self.smooth, accentuate=self.accentuate, thck=self.accentuate_thickness,
                                    replace_space=self.braille_space, space_strip=self.space_strip,
                                    threshold=(self.threshold_lower, self.threshold_upper))#, unihanCharacters=self.unihan_chars)
        char_count = len(ascii_img), max([len(line) for line in ascii_img.splitlines()]), ascii_img.count("\n")
        self.txtbox.insert("1.0", ascii_img)
        fdict = {
            4: (140, 430), 5: (120, 330), 6: (85, 325), 7: (70, 260), 8: (59, 210), 9: (56, 188), 10: (51, 188),
            11: (50, 155), 12: (48, 142), 13: (44, 135), 14: (41, 115), 15: (36, 119), 16: (35, 105)
        }
        self.txtbox.configure(height=fdict[self.font_size][0], width=fdict[self.font_size][1], font=("TkFixedFont", self.font_size))
        self.txt_count.configure(text=f"Characters: {char_count[0]}\nWidth (max): {char_count[1]}, Height: {char_count[2]}")
        self.font_label.configure(text=f"Font Size ({self.font_size})")

    def copy_to_clipboard(self):
        self.master.clipboard_clear()
        self.master.clipboard_append(self.txtbox.get("1.0", tk.END))


class ImageEditorWindow:
    def __init__(self, root, theme, console, transparent, unit, DPI, action_size):
        self.root = root
        self.unit = unit  # Units: px, cm, mm, in, pt --- 1px = 2.54/DPI cm = 25.4/DPI mm = 1/DPI in = DPI pt
        self.DPI = DPI
        self.file_path = ""
        self.cached_file_path = ""
        self.cached_paths = []
        self.animation_id = None
        self.zoom_animation_in_progress = False
        self.smooth_zoom = False
        self.im_num = 0
        self.theme = theme
        self.copy_image = True
        self.action_label = None
        self.after_id = None
        self.animation_thread = None

        if self.theme == "dark":
            self.bg_color = "#444444"
            self.button_color = "#343434"
            self.canvas_color = "#676767"
            self.text_color = "#FFFFFF"
            self.checker_color = "#898989", "#676767"
            self.cmd_color = "#121212"
            self.prompt_color = "#121212"
            self.text_color_cmd = "#FFFFFF"
        else:
            self.bg_color = "#ABABAB"
            self.button_color = "#EEEEEE"
            self.canvas_color = "#BCBCBC"
            self.text_color = "black"
            self.checker_color = "#EFEFEF", "#BCBCBC"
            self.cmd_color = "#DDDDDD"
            self.prompt_color = "#BBBBBB"
            self.text_color_cmd = "#000000"

        self.action_size = action_size
        self.current_action = -1
        self.action_list = []

        self.gif_maker_list, self.gif_maker_images = [], []
        self.changed_gif_maker = False
        self.gif_maker_frame = tk.Frame(root, height=100, width=root.winfo_reqwidth())
        self.gif_maker_canvas = tk.Canvas(self.gif_maker_frame, bg=self.bg_color, width=root.winfo_reqwidth(),
                                          height=130, scrollregion=(0, 0, 2000, 2000))
        self.gif_maker_hbar = tk.Scrollbar(self.gif_maker_frame, orient=tk.HORIZONTAL, command=self.gif_maker_canvas.xview)
        self.gif_maker_hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.gif_maker_hbar.bind("<B1-Motion>", lambda event: self.change_pan(val=False))
        self.gif_maker_hbar.bind("<ButtonRelease-1>", lambda event: self.change_pan(val=True))
        self.gif_maker_canvas.configure(xscrollcommand=self.gif_maker_hbar.set)
        self.gif_maker_canvas.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)

        self.command_inputs = []
        self.current = 0
        self.cmd_input = None
        self.cmd_output = None
        self.cmd = None

        self.left, self.right, self.top, self.bottom = 0, 0, 0, 0

        self.outline = False
        self.outline_color = (0, 0, 0, 255)
        self.picking_color = False
        self.using_wand = False
        self.painting = False
        self.keep_paint = False
        self.paintColor = [255, 0, 0, 50]
        self.paintMode = "alpha_compositing"
        self.brush = None
        self.brushSize = 15
        self.current_im_x, self.current_im_y = 0, 0
        self.wnd_tolerance = 20
        self.pressed_escape = False
        self.current_frame = 0
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.enable_pan = True
        self.can_pan = True
        self.button_frame = None
        self.psd_layers = None
        self.combo = None
        self.speed = 1.0
        self.in_other_window = False
        self.resize_method = Image.Resampling.NEAREST
        self.original_frames, self.original_delays = self.load_frames()
        self.delays = self.original_delays
        self.frames = None
        self.transparent = transparent
        self.drop_rectangle = None
        self.outline_threshold_select = None
        self.outline_slider = None
        self.color_picked = [(255, 255, 255), (255, 255, 255)]
        self.dnd_image = ImageTk.PhotoImage(Image.open(r'gui_images/dnd2.png').resize((150, 150)))
        self.colpickimg = ImageTk.PhotoImage(Image.open(r"gui_images/color_picker.png").resize((60, 60)))
        self.wandimg = ImageTk.PhotoImage(Image.open(r"gui_images/magic_wand.png").resize((60, 60)))
        self.create_widgets()

        root.bind("<Button-1>", self.start_pan)
        root.bind("<ButtonRelease-1>", self.stop_pan)
        root.bind("<B1-Motion>", self.pan)
        root.bind("<KeyRelease-space>", self.pause)
        root.bind("<Left>", self.prev_frame)
        root.bind("<Right>", self.next_frame)
        root.bind("<Shift-plus>", lambda event: self.gif_maker(type="add"))
        root.bind("<Shift-minus>", lambda event:  self.gif_maker(type="last"))
        root.bind("<Control-Shift-minus>", lambda event: self.gif_maker(type="first"))
        root.bind("<Alt-minus>", lambda event: self.gif_maker(type="clear_all"))
        root.bind("<Shift-G>", lambda event: self.gif_maker(type="create"))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', background=self.button_color, fieldbackground=self.button_color,
                        foreground=self.text_color, font=('Verdana', 10), padding=(10, 5),
                        darkcolor=self.bg_color, lightcolor=self.bg_color)

        self.options = {
            'Activate Console': console,
            'Use Optimizations': True,
            'Show FPS': False,
            'Smooth Zoom': self.smooth_zoom,
            'Copy Path': not self.copy_image
        }
        self.apply_options(self.options, None, None)
        self.optimizations = True

        root.bind("<Control-Alt-f>", self.open_options_window)
        root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
        root.bind("<F12>", self.open_cmd)
        root.bind("<Control-o>", lambda event: self.change_path())
        root.bind("<Control-s>", lambda event: self.save_image())
        root.bind("<Control-n>", lambda event: self.new_image())
        root.bind('<Escape>', lambda event: self.on_exit())
        root.bind("<Control-Alt-t>", lambda event: self.change_theme())
        root.bind("<Control-Alt-c>", lambda event: self.quality())
        root.bind("<Control-c>", lambda event: self.copy_current_image())
        root.bind("<Control-v>", lambda event: self.paste_current_image())
        root.bind("<Control-z>", lambda event: self.undo())
        root.bind("<Control-y>", lambda event: self.redo())
        root.bind("<i>", lambda event: self.color_picker(True))
        root.bind("<w>", lambda event: self.magic_wand(True))
        root.bind("<Delete>", lambda event: self.supr())

        root.bind("<p>", lambda event: self.paint_brush())
        root.bind("<Up>", lambda event: self.brush_size_up())
        root.bind("<Down>", lambda event: self.brush_size_down())

        root.bind("<Control-Shift-F>", lambda event: self.filter_window())
        root.bind("<Control-Shift-K>", lambda event: self.kernel_text())
        root.bind("<Control-Shift-R>", lambda event: self.resize_image())
        root.bind("<Control-Shift-O>", lambda event: self.outline_image())
        root.bind("<Control-Shift-N>", lambda event: self.add_noise())
        root.bind("<Control-Shift-A>", lambda event: self.convert_ascii())

        root.minsize(426, 240)
        root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.fps = False
        self.fps_label = None
        self.show_fps()

    def change_pan(self, event=None, val=True):
        self.can_pan = val

    def gif_maker(self, event=None, type="add"):
        if not self.frames:
            action = 'delete frame'
            if type == "create": action = 'create gif'
            elif type == "add": action = 'add frame'
            self.show_action_label(
                f"Couldn't {action}: No image loaded.")
            return
        if type == "create":
            if not self.gif_maker_list:
                self.show_action_label("Couldn't create gif: No frames added.\nUse Shift+plus to add the current image frame.")
                return
            delays = [round(60/self.speed) for _ in self.gif_maker_list]
            gif_frames = []
            for i, frame in enumerate(self.gif_maker_list):
                frame = frame.convert("RGBA")
                alpha = frame.split()[3]
                new_frame = Image.new("RGBA", frame.size, (0, 0, 0, 0))
                new_frame.paste(frame, (0, 0), mask=alpha)
                gif_frames.append(new_frame)
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            os.makedirs(folder_path, exist_ok=True)
            self.im_num += 1
            self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}.gif"
            gif_frames[0].save(self.cached_file_path, save_all=True, append_images=gif_frames[1:], duration=delays, loop=0,
                               disposal=2)
            if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
            self.action_list.append([self.gif_maker_list, self.cached_file_path])
            if len(self.action_list) > self.action_size:
                self.action_list.pop(0)
            self.current_action = len(self.action_list) - 1
            self.file_path = self.cached_file_path
            self.current_frame = 0
            self.original_frames, self.delays = self.load_frames()
            self.frames = self.original_frames.copy()
            self.show_action_label("Gif created. Ctrl+S to save it.")
        elif type == "last" and self.gif_maker_list: self.gif_maker_list.pop()
        elif type == "first" and self.gif_maker_list: self.gif_maker_list.pop(0)
        elif type == "add": self.gif_maker_list.append(self.action_list[self.current_action][0][self.current_frame])
        elif type == "clear_all": self.gif_maker_list.clear()
        self.changed_gif_maker = True
        self.update_frame()

    def change_unit(self, unit):
        if self.outline:
            self.outline_image(False)
        self.unit = unit
        self.show_action_label(f"Unit changed to `{unit}`")

    def convert_unit(self, value): # convert from px to the selected unit
        if self.unit == "cm": value *= 2.54 / self.DPI
        elif self.unit == "mm": value *= 25.4 / self.DPI
        elif self.unit == "in": value /= self.DPI
        elif self.unit == "pt": value *= 72 / self.DPI
        return value

    def convert_back(self, value): # convert from the selected unit to px
        if self.unit == "cm": value *= self.DPI / 2.54
        elif self.unit == "mm": value *= self.DPI / 25.4
        elif self.unit == "in": value *= self.DPI
        elif self.unit == "pt": value *= self.DPI / 72
        return round(value)

    def copy_current_image(self):
        if not self.frames: return
        self.root.clipboard_clear()

        if self.copy_image:
            img = self.action_list[self.current_action][0][self.current_frame]
            img = img.convert('RGBA')

            output = io.BytesIO()
            img.save(output, format="BMP")
            image_data = output.getvalue()[14:]
            output.close()

            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIBV5, image_data)
            win32clipboard.CloseClipboard()
            self.show_action_label("Copied current image to clipboard.")
            return
        self.root.clipboard_append(self.cached_paths[self.current_action])
        self.root.update()
        self.show_action_label("Copied current image cached path to clipboard.")

    def paste_current_image(self):
        try:
            self.change_image(self.root.clipboard_get())
            self.show_action_label("Pasted image.")
        except:
            try:
                folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
                os.makedirs(folder_path, exist_ok=True)
                self.im_num += 1
                self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}.png"
                ImageGrab.grabclipboard().save(self.cached_file_path)
                if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
                self.change_image(self.cached_file_path)
                self.show_action_label("Pasted image.")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", f"Failed to paste file: {e}")

    def undo(self, limit=""):
        if limit != "":
            self.stop()
            self.action_size = int(limit)
        else:
            if self.current_action < 0: return None
            self.stop()
            self.current_action = self.current_action - 1 if self.current_action > 0 else 0
            self.frames = self.original_frames = self.action_list[self.current_action][0]
            pathpsd = self.action_list[self.current_action][1].lower().endswith('.psd')
            if pathpsd and not self.combo:
                layers = [layer.name for layer in self.psd_layers]
                layers.append("All Layers")

                self.combo = ttk.Combobox(self.root, values=layers, style='TCombobox')
                self.combo.pack(pady=10)
                self.combo.set(layers[-1])
                self.combo.bind("<<ComboboxSelected>>", self.update_displayed_layer)
                self.change_theme(chng=self.theme)
            elif not pathpsd and self.combo:
                self.combo.destroy()
                self.combo = None
        self.update_frame()
        self.play()
        pass

    def redo(self, limit=""):
        if limit != "":
            self.stop()
            self.action_size = int(limit)
        else:
            if self.current_action < 0: return None
            self.stop()
            self.current_action = self.current_action + 1 if self.current_action < len(self.action_list)-1 else len(self.action_list)-1
            self.frames = self.original_frames = self.action_list[self.current_action][0]
            pathpsd = self.action_list[self.current_action][1].lower().endswith('.psd')
            if pathpsd and not self.combo:
                layers = [layer.name for layer in self.psd_layers]
                layers.append("All Layers")

                self.combo = ttk.Combobox(self.root, values=layers, style='TCombobox')
                self.combo.pack(pady=10)
                self.combo.set(layers[-1])
                self.combo.bind("<<ComboboxSelected>>", self.update_displayed_layer)
                self.change_theme(chng=self.theme)
            elif not pathpsd and self.combo:
                self.combo.destroy()
                self.combo = None
        self.update_frame()
        self.play()
        pass

    def on_exit(self):
        if self.picking_color:
            self.color_picker()
            self.pressed_escape = True
            self.update_frame()
        elif self.using_wand:
            self.magic_wand()
            self.pressed_escape = True
            self.update_frame()
        elif messagebox.askyesno("Exit", "Do you want to exit?"):
            self.delete_cache(all=True)
            self.root.destroy()

    def update_fps(self, state="disable"):
        start_time = time.time()

        self.update_frame()

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        if self.fps:
            self.fps_label.config(text=f"FPS: {fps:.2f}")
            self.root.after(500, self.update_fps)

    def load_frames(self):
        if self.combo:
            self.combo.destroy()
            self.combo = None
        if self.file_path == "": return [], []
        if self.file_path.lower().endswith(tuple(FILETYPES[1:])):
            if self.file_path.lower().endswith('.psd'):
                self.psd_layers = PSDImage.open(self.file_path)

                layers = [layer.name for layer in self.psd_layers]
                layers.append("All Layers")

                self.combo = ttk.Combobox(self.root, values=layers, style='TCombobox')
                self.combo.pack(pady=10)
                self.combo.set(layers[-1])
                self.combo.bind("<<ComboboxSelected>>", self.update_displayed_layer)

                frames = [self.psd_layers.composite()]
                delays = [0]
            else:
                frames = [Image.open(self.file_path)]
                delays = [0]
        else:
            gif = Image.open(self.file_path)
            frames = []
            delays = []

            try:
                while True:
                    frames.append(gif.copy().convert("RGBA"))
                    delays.append(gif.info['duration'])
                    non_zero = list(filter(lambda x: x != 0, delays))
                    min_value = min(non_zero) if non_zero else 20
                    delays = [min_value if delay == 0 else delay for delay in delays]
                    gif.seek(len(frames))
            except EOFError:
                pass

            print(f"\033[94mDelays for each frame (ms): {';'.join(str(delay) for delay in delays)}\033[0m")
            print(f"\033[94mFrames: {len(frames)}\033[0m")
            self.display_output(f"Delays for each frame (ms): {';'.join(str(delay) for delay in delays)}")
            self.display_output(f"Frames: {len(frames)}")

        print("\033[92mImage loaded.\033[0m\n" if self.file_path.lower().endswith(tuple(FILETYPES[1:])) else "\033[92mGif loaded.\033[0m\n")
        self.display_output("Image loaded.\n" if self.file_path.lower().endswith(tuple(FILETYPES[1:])) else "Gif loaded.\n")
        self.show_action_label("Image loaded." if self.file_path.lower().endswith(tuple(FILETYPES[1:])) else "Gif loaded.")
        return frames, delays

    def load_url(self):
        try:
            url = simpledialog.askstring("Load from URL", "Enter the URL of the image:")
            response = requests.get(url)
            if response.status_code == 200:
                #frames = response.content.decode('utf-8').split(';')
                #for i in range(len(frames)):
                #    frames[i] = Image.open(BytesIO(frames[i].encode('utf-8')))
                frames = [Image.open(BytesIO(response.content))]

                self.original_frames = self.frames = [frame for frame in frames]
                self.original_delays = self.delays = [40] * len(frames)
                self.file_path = "image.png"

                folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
                os.makedirs(folder_path, exist_ok=True)
                self.im_num += 1
                self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
                if not self.file_path.endswith(('.psd', '.cur')):
                    self.save_image(self.cached_file_path)
                    if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
                self.action_list.append([self.frames, self.file_path])
                if len(self.action_list) > self.action_size:
                    self.action_list.pop(0)
                self.current_action = len(self.action_list) - 1
                self.show_action_label("URL Loaded.")

            else:
                print(f"Failed to fetch image from URL. Status code: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image from URL: {e}")
            traceback.print_exc()
            return None
        self.update_frame()

    def create_widgets(self):
        sep = 1
        self.buttons = {}
        self.menus = {}

        self.button_frame = tk.Frame(self.root, bg=self.bg_color)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        load_image_frame = tk.Frame(self.button_frame, bg=self.bg_color)
        load_image_frame.pack(side=tk.LEFT, padx=10)

        self.change_path_button = tk.Button(load_image_frame, text="Load Image", command=self.change_path)
        self.change_path_button.pack(side=tk.LEFT)
        self.buttons["change_path_button"] = self.change_path_button

        self.load_url_button = tk.Button(load_image_frame, text="Load from URL", command=self.load_url)
        self.load_url_button.pack(side=tk.LEFT)
        self.buttons["load_url_button"] = self.load_url_button

        filter_kernel_frame = tk.Frame(self.button_frame, bg=self.bg_color)
        filter_kernel_frame.pack(side=tk.LEFT, padx=10)

        self.apply_filter_button = tk.Button(filter_kernel_frame, text="Filter", command=self.filter_window)
        self.apply_filter_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["apply_filter_button"] = self.apply_filter_button

        self.apply_kernel_button = tk.Button(filter_kernel_frame, text="Kernel", command=self.kernel_text)
        self.apply_kernel_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["apply_kernel_button"] = self.apply_kernel_button

        self.outline_button = tk.Button(filter_kernel_frame, text="Outline", command=self.outline_image)
        self.outline_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["outline_button"] = self.outline_button

        self.resize_button = tk.Button(filter_kernel_frame, text="Resize", command=self.resize_image)
        self.resize_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["resize_button"] = self.resize_button

        play_pause_stop_frame = tk.Frame(self.button_frame, bg=self.bg_color)
        play_pause_stop_frame.pack(side=tk.LEFT, padx=10)

        self.play_button = tk.Button(play_pause_stop_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["play_button"] = self.play_button

        self.pause_button = tk.Button(play_pause_stop_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["pause_button"] = self.pause_button

        self.stop_button = tk.Button(play_pause_stop_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["stop_button"] = self.stop_button

        apply_save_frame = tk.Frame(self.button_frame, bg=self.bg_color)
        apply_save_frame.pack(side=tk.LEFT, padx=10)

        self.apply_changes_button = tk.Button(apply_save_frame, text="Apply Changes", command=self.apply_changes)
        self.apply_changes_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["apply_changes_button"] = self.apply_changes_button

        self.save_image_button = tk.Button(apply_save_frame, text="Save Image", command=self.save_image)
        self.save_image_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["save_image_button"] = self.save_image_button

        misc_frame = tk.Frame(self.button_frame, bg=self.bg_color)
        misc_frame.pack(side=tk.LEFT, padx=10)

        self.change_theme_button = tk.Button(misc_frame, text="Change Theme", command=self.change_theme)
        self.change_theme_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["change_theme_button"] = self.change_theme_button

        self.enable_checker_button = tk.Button(misc_frame, text="Transparent BG", command=self.enable_checker)
        self.enable_checker_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["enable_checker_button"] = self.enable_checker_button

        self.reset_view_button = tk.Button(misc_frame, text="Reset View", command=self.reset_view)
        self.reset_view_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["reset_view_button"] = self.reset_view_button

        self.change_quality_button = tk.Button(misc_frame, text="Change Quality", command=self.quality)
        self.change_quality_button.pack(side=tk.LEFT, padx=sep)
        self.buttons["change_quality_button"] = self.change_quality_button

        self.color_picker_button = tk.Button(self.root, text="ColPick", image=self.colpickimg, command=self.color_picker)
        self.buttons["color_picker_button"] = self.color_picker_button

        self.magic_wand_button = tk.Button(self.root, text="MgcWnd", image=self.wandimg, command=self.magic_wand)
        self.buttons["magic_wand_button"] = self.magic_wand_button

        for button in self.buttons:
            if (button == "cmd_button" and not self.cmd) or ():
                pass
            else:
                self.buttons[button].config(bg=self.button_color, fg=self.text_color,
                                            activebackground=self.button_color, activeforeground=self.text_color,
                                            font=('Verdana', 10), bd=0, cursor="hand2", highlightthickness=3)

        if self.file_path:
            size = get_image_size(self.file_path)
            _, file_extension = os.path.splitext(self.file_path)
        else:
            size = (0, 0)
            file_extension = ''
        self.info_label = tk.Label(self.root, text=f"Image info: No image loaded."
                                       f" Zoom method: {str(self.resize_method).replace('Resampling.', '')}. Current image: 0/0.",
                                   bg=self.bg_color, fg=self.text_color, font=('Verdana', 10))
        self.info_label.pack(side=tk.BOTTOM, anchor=tk.W, fill=tk.X)

        self.canvas = tk.Canvas(self.root, bg=self.canvas_color, width=1280, height=720, borderwidth=2, highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.update_image_position)
        self.x1, self.y1, self.x2, self.y2 = 50, 50, self.canvas.winfo_reqwidth() - 100, self.canvas.winfo_reqheight() - 100

        self.draw_checkered_pattern(self.checker_color)

        self.menu_bar = tk.Menu(self.root)
        self.menus["menu_bar"] = self.menu_bar
        self.root.config(menu=self.menu_bar)

        # "FILE" MENU
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New", command=self.new_image, accelerator="Ctrl+N")
        self.file_menu.add_command(label="Open", command=self.change_path, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Save", command=self.save_image, accelerator="Ctrl+S")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Copy", command=self.copy_current_image, accelerator="Ctrl+C")
        self.file_menu.add_command(label="Paste", command=self.paste_current_image, accelerator="Ctrl+V")
        self.file_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        self.file_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.destroy, accelerator="Esc")
        self.menus["file_menu"] = self.file_menu

        # "PREFERENCES" MENU
        self.preferences_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.theme_menu = tk.Menu(self.preferences_menu, tearoff=0)
        self.quality_menu = tk.Menu(self.preferences_menu, tearoff=0)
        self.units_menu = tk.Menu(self.preferences_menu, tearoff=0)
        self.menu_bar.add_cascade(label="Preferences", menu=self.preferences_menu)
        self.preferences_menu.add_cascade(label="Theme", menu=self.theme_menu)
        self.theme_menu.add_command(label="Dark", command=lambda: self.change_theme("dark"))
        self.theme_menu.add_command(label="Light", command=lambda: self.change_theme("light"))
        self.preferences_menu.add_cascade(label="Quality", menu=self.quality_menu)
        self.quality_menu.add_command(label="High", command=lambda: self.quality(Image.Resampling.BICUBIC))
        self.quality_menu.add_command(label="Low", command=lambda: self.quality(Image.Resampling.NEAREST))
        self.preferences_menu.add_command(label="Fullscreen", command=lambda: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")), accelerator="F11")
        self.preferences_menu.add_cascade(label="Units", menu=self.units_menu)
        self.units_menu.add_command(label="Pixels", command=lambda: self.change_unit("px"))
        self.units_menu.add_command(label="Inches", command=lambda: self.change_unit("in"))
        self.units_menu.add_command(label="Centimeters", command=lambda: self.change_unit("cm"))
        self.units_menu.add_command(label="Milimeters", command=lambda: self.change_unit("mm"))
        self.units_menu.add_command(label="Points", command=lambda: self.change_unit("pt"))

        self.menus["preferences_menu"] = self.preferences_menu
        self.menus["theme_menu"] = self.theme_menu
        self.menus["quality_menu"] = self.quality_menu

        # "EDIT" MENU
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)
        self.edit_menu.add_command(label="Filter", command=self.filter_window, accelerator="Ctrl+Shift+F")
        self.edit_menu.add_command(label="Kernel", command=self.kernel_text, accelerator="Ctrl+Shift+K")
        self.edit_menu.add_command(label="Resize", command=self.resize_image, accelerator="Ctrl+Shift+R")
        self.edit_menu.add_command(label="Outline", command=self.outline_image, accelerator="Ctrl+Shift+O")
        self.edit_menu.add_command(label="Add Noise", command=self.add_noise, accelerator="Ctrl+Shift+N")
        self.edit_menu.add_command(label="Img to ASCII", command=self.convert_ascii, accelerator="Ctrl+Shift+A")
        self.menus["edit_menu"] = self.edit_menu

        # "ANIMATION" MENU
        self.animation_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Animation", menu=self.animation_menu)
        self.animation_menu.add_command(label="Play", command=self.play, accelerator="Space")
        self.animation_menu.add_command(label="Pause", command=self.pause, accelerator="Space")
        self.animation_menu.add_command(label="Stop", command=self.stop)
        self.animation_menu.add_command(label="Next Frame", command=self.next_frame, accelerator="Right Arrow")
        self.animation_menu.add_command(label="Previous Frame", command=self.prev_frame, accelerator="Left Arrow")
        self.menus["animation_menu"] = self.animation_menu

        # "TOOLS" MENU
        self.tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=self.tools_menu)
        self.tools_menu.add_command(label="Color Pick/Eyedropper", command=lambda: self.color_picker(True), accelerator="I")
        self.tools_menu.add_command(label="Magic Wand", command=lambda: self.magic_wand(True), accelerator="W")
        self.menus["tools_menu"] = self.tools_menu

        # "COMMAND" MENU
        self.command_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Command", menu=self.command_menu)
        self.command_menu.add_command(label="Run", command=self.window_cmd, accelerator="F12")
        self.command_menu.add_command(label="Delete Cache", command=self.delete_cache)
        self.menus["command_menu"] = self.command_menu

        for menu in self.menus:
            self.menus[menu].config(bg=self.button_color, fg=self.text_color,
                                   activeforeground=self.text_color, font=('Verdana', 10), borderwidth=0)

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)
        self.root.dnd_bind('<<DropEnter>>', self.on_drag_enter)
        self.root.dnd_bind('<<DropLeave>>', self.on_drag_leave)

        self.root.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<+>", self.zoom_in)
        self.root.bind("-", self.zoom_out)
        self.root.bind("<Configure>", self.on_window_resize)

        self.color_picker_button.place(x=self.canvas.winfo_reqwidth() - 75, y=self.canvas.winfo_reqheight() - 100)
        self.color_picker_button.lift()
        self.magic_wand_button.place(x=self.canvas.winfo_reqwidth() - 75, y=self.canvas.winfo_reqheight() - 170)
        self.magic_wand_button.lift()

    def show_action_label(self, text, duration=5000):
        # called each time an action is done (filtering, kernel, etc) with info
        if not self.action_label:
            self.action_label = tk.Label(self.root, text=text, bg=self.button_color, fg=self.text_color, font=("Verdana", 16))
            self.action_label.place(x=10, y=40)
        else:
            self.action_label.configure(text=text)
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(duration, self.remove_action_label)

    def remove_action_label(self):
        if self.action_label: self.action_label.destroy()
        self.action_label = None
        self.after_id = None

    def supr(self):
        if self.using_wand:
            if messagebox.askyesno("Erase", "Erase selected region?"):
                def threads_wand(frame):
                    return Image.fromarray(im.magicWand(np.array(frame.convert("RGBA")), (self.current_im_x, self.current_im_y), (0, 0, 0, 0), self.wnd_tolerance))

                with ThreadPoolExecutor(max_workers=6) as executor:
                    self.frames = list(executor.map(threads_wand, self.original_frames))

                #self.frames = [Image.fromarray(im.magicWand(np.array(frame.convert("RGBA")), (self.current_im_x, self.current_im_y), (0, 0, 0, 0), self.wnd_tolerance)) for frame in self.original_frames]
                self.original_frames = self.frames.copy()

                folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
                os.makedirs(folder_path, exist_ok=True)
                self.im_num += 1
                self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
                if not self.file_path.endswith(('.psd', '.cur')):
                    self.save_image(self.cached_file_path)
                    if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
                self.action_list.append([self.frames, self.file_path])
                if len(self.action_list) > self.action_size:
                    self.action_list.pop(0)
                self.current_action = len(self.action_list) - 1
            self.update_frame()


    def pick_color(self, event):
        x, y = self.root.winfo_pointerxy()
        screenshot = pyautogui.screenshot()
        color = screenshot.getpixel((x, y))
        self.color_picked = [self.color_picked[1], color]
        self.update_frame()

    def pick_wand(self, event):
        fx, fy = self.frames[0].size
        ww, wh = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = int(self.canvas.winfo_pointerx()-self.canvas.winfo_rootx()-self.pan_offset_x-(ww-fx)//2),\
               int(self.canvas.winfo_pointery()-self.canvas.winfo_rooty()-self.pan_offset_y-(wh-fy)//2)
        print(x, y)
        color = np.array(self.frames[0].convert("RGBA"))[y, x]
        color[3] = 60
        if 0 <= x <= fx and 0 <= y <= fy:
            def threads_wand(frame):
                return Image.fromarray(im.magicWand(np.array(frame.convert("RGBA")), (x, y), tuple(color), self.wnd_tolerance))

            with ThreadPoolExecutor(max_workers=6) as executor:
                self.frames = list(executor.map(threads_wand, self.original_frames))
            #self.frames = [Image.fromarray(im.magicWand(np.array(frame.convert("RGBA")), (x, y), tuple(color), self.wnd_tolerance)) for frame in self.original_frames]
            self.current_im_x, self.current_im_y = x, y
        self.update_frame()

    def get_brush(self, brushType='circle'):
        brushThickness = 2 * self.brushSize - 1
        thck_tuple = (brushThickness, brushThickness)
        brushKernel = np.ones(thck_tuple)
        brushType = brushType.lower()
        hardness = 2
        hardness = min(max(hardness, 0.0), 2.0)

        if brushType == "circle":
            center = (brushThickness - 1) / 2
            radius = brushThickness // 2

            indices = np.indices(thck_tuple)
            distances = np.sqrt((indices[0] - center) ** 2 + (indices[1] - center) ** 2)

            weight = 1.0 - hardness * distances / (radius + np.sqrt(center / (32 * radius)))
            brushKernel = weight * (distances <= radius + np.sqrt(center / (32 * radius)))
        elif brushType == "star":
            indices = np.abs(np.arange(-brushThickness // 2 + 1, brushThickness // 2 + 1))
            brushKernel = (indices[:, np.newaxis] + indices[np.newaxis, :]) <= brushThickness // 2
        return np.clip(brushKernel.astype(float), 0, 1)

    def brush_size_up(self):
        self.brushSize += 1

    def brush_size_down(self):
        if self.brushSize <= 1: self.brushSize = 1
        self.brushSize -= 1

    def paint(self, event=None, brushKernel=None):
        if self.keep_paint:
            fx, fy = self.frames[0].size
            ww, wh = self.canvas.winfo_width(), self.canvas.winfo_height()
            x, y = int(self.canvas.winfo_pointerx()-self.canvas.winfo_rootx()-self.pan_offset_x-(ww-fx)//2),\
                   int(self.canvas.winfo_pointery()-self.canvas.winfo_rooty()-self.pan_offset_y-(wh-fy)//2)
            if not brushKernel:
                self.brush = self.get_brush()
                brushKernel = self.brush
            if 0 <= x <= fx and 0 <= y <= fy:
                self.frames = [Image.fromarray(im.paint(np.array(frame.convert("RGBA")), (y, x), brushKernel, self.paintColor, self.paintMode)) for frame in self.frames]
            self.update_frame()
            self.root.after(10, self.paint)
        self.keep_paint = True

    def paint_brush(self):
        if self.using_wand or self.picking_color: return
        self.painting = not self.painting
        if self.painting:
            self.can_pan = False
            self.keep_paint = True
            self.root.config(cursor="@gui_images/magicwndd.cur")
            self.root.bind("<ButtonPress-1>", self.paint)
        else:
            self.can_pan = True
            self.root.config(cursor="")
            self.root.unbind("<ButtonPress-1>")
            self.root.bind("<ButtonPress-1>", self.start_pan)
            self.frames = self.original_frames.copy()

    def color_picker(self, p=False):
        if self.using_wand or self.painting: return
        self.picking_color = not self.picking_color
        self.pressed_escape = p
        if self.picking_color:
            self.can_pan = False
            self.root.config(cursor="@gui_images/eyedropper.cur")
            self.root.bind("<Button-1>", self.pick_color)
        else:
            self.can_pan = True
            self.root.config(cursor="")
            self.root.unbind("<Button-1>")
            self.root.bind("<Button-1>", self.start_pan)
        self.update_frame()

    def change_crop(self, event=None, type="left"):
        if type == "left": self.left = int(self.left_slider.get())
        elif type == "right": self.right = int(self.right_slider.get())
        elif type == "top": self.top = int(self.top_slider.get())
        elif type == "bottom": self.bottom = int(self.bottom_slider.get())
        for i in range(len(self.original_frames)):
            frame = np.array(self.original_frames[i].copy())
            self.frames[i] = Image.fromarray(frame[self.top:frame.shape[0] - self.bottom, self.left:frame.shape[1] - self.right])

    def magic_wand(self, p=False):
        if self.picking_color or self.painting: return
        self.using_wand = not self.using_wand
        self.pressed_escape = p
        if self.using_wand:
            self.can_pan = False
            self.root.config(cursor="@gui_images/magicwndd.cur")
            self.root.bind("<Button-1>", self.pick_wand)
        else:
            self.can_pan = True
            self.root.config(cursor="")
            self.root.unbind("<Button-1>")
            self.root.bind("<Button-1>", self.start_pan)
            self.frames = self.original_frames.copy()
        self.update_frame()

    def outline_image(self, apply=True):
        if not self.file_path: return None
        self.outline = not self.outline
        if self.outline:
            # LABELS AND LINES
            self.thck_label = tk.Label(self.root, text=f"Thickness ({self.unit})", bg=self.bg_color, fg=self.text_color, font=('Verdana', 12))
            self.type_label = tk.Label(self.root, text="Outline Type", bg=self.bg_color, fg=self.text_color, font=('Verdana', 12))
            self.thr_label = tk.Label(self.root, text="Threshold", bg=self.bg_color, fg=self.text_color, font=('Verdana', 12))
            self.empty_label = tk.Label(self.root, text="", bg=self.bg_color, fg=self.text_color)
            self.empty_label2 = tk.Label(self.root, text="", bg=self.bg_color, fg=self.text_color)
            self.thck_label.place(x=0, y=30, relwidth=0.25, height=30)
            self.type_label.place(relx=0.25, y=30, relwidth=0.25, height=30)
            self.empty_label.place(relx=0.5, y=30, relwidth=0.25, height=30)
            self.thr_label.place(relx=0.75, y=30, relwidth=0.25, height=30)

            if not self.outline_threshold_select: thr_value = 50
            else: thr_value = self.outline_threshold_select.get()

            # SLIDER AND BUTTONS
            self.outline_slider = tk.Scale(self.root, from_=0, to=self.convert_unit(int(min(self.frames[0].width, self.frames[0].height) / 8)),
                                           orient="horizontal", command=lambda val: self.change_outline(val, thr_value), font=('Verdana', 12),
                                           bd=0, highlightthickness=0, resolution=self.convert_unit(1))

            self.outline_type = ttk.Combobox(self.root, values=["Circle", "Square", "Star"], font=('Verdana', 12), style='TCombobox')
            self.outline_type.set("Circle")
            self.outline_type.bind("<<ComboboxSelected>>", self.on_type_change)

            self.outline_color_select = tk.Button(self.root, text="Choose Outline Color", command=self.choose_color, font=('Verdana', 12))
            self.outline_threshold_select = tk.Scale(self.root, from_=0, to=255, orient="horizontal",
                            command=lambda th: self.change_outline(self.outline_slider.get(), th), font=('Verdana', 12),
                                           bd=0, highlightthickness=0)

            self.outline_threshold_select.set(50)
            h = self.outline_slider.winfo_reqheight()
            self.outline_slider.place(x=0, y=60, relwidth=0.25, height=h)
            self.outline_type.place(relx=0.25, y=60, relwidth=0.25, height=h)
            self.outline_color_select.place(relx=0.5, y=60, relwidth=0.25, height=h)
            self.outline_threshold_select.place(relx=0.75, y=60, relwidth=0.25,height=h)
            self.empty_label2.place(x=0, y=h+60, relwidth=1, height=15)
            self.change_theme(chng=self.theme)
        else:
            if apply: self.apply_changes()
            self.outline_slider.destroy()
            self.outline_type.destroy()
            self.outline_color_select.destroy()
            self.outline_threshold_select.destroy()
            self.thck_label.destroy()
            self.type_label.destroy()
            self.thr_label.destroy()
            self.empty_label.destroy()
            self.empty_label2.destroy()
            self.change_outline(0)
            self.outline_threshold_select = None
            self.outline_slider = None
        self.update_frame()

    def on_type_change(self, event):
        if not self.outline_threshold_select: thr_value = 50
        else: thr_value = self.outline_threshold_select.get()
        if not self.outline_slider: val_value = 0
        else: val_value = self.outline_slider.get()
        self.change_outline(val_value, thr_value)

    def choose_color(self):
        rgb = colorchooser.askcolor(title="Choose a color", color=(0, 0, 0))[0]
        if not rgb: return None
        self.outline_color = (*rgb, 255)
        self.change_outline(self.outline_slider.get())

    def change_outline(self, value, thr=50):
        if not self.file_path: return None
        value = self.convert_back(float(value))
        if value == 0:
            self.frames = self.original_frames.copy()
            return None
        self.enable_pan = False
        value = 2*int(value)-1

        self.frames = [Image.fromarray(im.imageOutline(np.array(frame.convert("RGBA")), thickness=value, threshold=int(thr), outlineColor=self.outline_color, outlineType=self.outline_type.get().lower())) for frame in self.original_frames]
        self.update_frame()

    def new_image(self):
        self.file_path = ""
        self.cached_file_path = ""
        self.animation_id = None
        self.command_inputs = []
        self.original_frames, self.original_delays = self.load_frames()
        self.frames, self.delays = self.load_frames()
        self.update_frame()
        self.show_action_label("Canvas emptied.")

    def resize_image(self, args=[]):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        orig_size = get_image_size(self.file_path)
        if not args:
            self.in_other_window = True
            dialog = ResizeWindow(self.root, orig_size, self.theme)
            self.in_other_window = False
            size = (dialog.width, dialog.height)
            res_method = dialog.resize_method
            if res_method.lower() == "bilinear": method = Image.Resampling.BILINEAR
            elif res_method.lower() == "bicubic": method = Image.Resampling.BICUBIC
            else: method = Image.Resampling.NEAREST
        else:
            new_w = int(args[0])
            new_h = int(args[1])
            method = Image.Resampling.BICUBIC
            if len(args) == 3:
                args[2] = convert_to_type(args[2])
                if args[2]:
                    aspect_ratio = int(orig_size[1]) / int(orig_size[0])
                    new_h = int(new_w * aspect_ratio)
            elif len(args) > 3:
                args[2] = convert_to_type(args[2])
                if args[2]:
                    aspect_ratio = int(orig_size[1]) / int(orig_size[0])
                    new_h = int(new_w * aspect_ratio)
                if args[3].lower() == "bilinear": method = Image.Resampling.BILINEAR
                elif args[3].lower() == "bicubic": method = Image.Resampling.BICUBIC
                else: method = Image.Resampling.NEAREST
            size = (new_w, new_h)
        if not all(size): return None
        new_images = np.zeros(len(self.frames)).tolist()
        for i in range(len(self.frames)):
            new_images[i] = self.original_frames[i].resize(size, method)
        self.frames = copy.deepcopy(new_images)
        self.update_frame()
        self.action_list.append([self.frames, self.file_path])
        if len(self.action_list) > self.action_size:
            self.action_list.pop(0)
        self.current_action = len(self.action_list) - 1
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(folder_path, exist_ok=True)
        self.im_num += 1
        self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
        if not self.file_path.endswith(('.psd', '.cur')):
            self.save_image(self.cached_file_path)
            if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
        self.update_frame()
        self.show_action_label(f"Resized image to {size[0]}x{size[1]}")

    def filter_window(self, args=[]):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        if not args:
            self.in_other_window = True
            dialog = FilterWindow(self.root, self.theme, self.frames)
            self.in_other_window = False
            ftype = dialog.filter_type
            fpix = dialog.pixelization
            fcustom = dialog.custom_filter
            fpalette = dialog.palette
            fonepal, fprepal = dialog.one_palette, dialog.pre_palette
            if isinstance(dialog.palette, str): fpalette = fpalette.lower()
            else: fpalette = None
            fmirror, fgreycol = dialog.mirror, dialog.colors
            fdittype, fditcol = dialog.dither_type, int(4 if dialog.dither_colors == "" else dialog.dither_colors)
            fsat, fcon, flum = dialog.saturation, dialog.contrast, dialog.luminance
        else:
            ftype = args[0]
            fpix = args[1]
        if not fpalette and fonepal and fdittype:
            fpalette = np.array(fprepal).copy()
        param_dict = {
            "pixelSize": fpix, "saturation": fsat, "contrast": fcon, "luminance": flum,
            "customFilter": fcustom, "mirror": fmirror, "colors": fgreycol,
            "ditherType": fdittype, "ditherColors": fditcol, "customPalette": fpalette,
        }
        ftype = ftype.lower()

        def threads_filter(frame):
            return Image.fromarray(im.imageFilter(np.array(frame.convert("RGBA")), ftype, **param_dict))

        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                self.frames = list(executor.map(threads_filter, self.original_frames))
            #self.frames = [Image.fromarray(im.imageFilter(np.array(frame.convert("RGBA")), ftype, **param_dict)) for frame in self.original_frames]
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            traceback.print_exc()
            return None
        self.current_frame = 0

        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(folder_path, exist_ok=True)
        self.im_num += 1
        self.cached_file_path = folder_path+f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
        if not self.file_path.endswith(('.psd', '.cur')):
            self.save_image(self.cached_file_path)
            if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
        self.action_list.append([self.frames, self.file_path])
        if len(self.action_list) > self.action_size:
            self.action_list.pop(0)
        self.current_action = len(self.action_list) - 1
        self.update_frame()
        text = "Filters applied:\n"
        text += 'No filter\n' if ftype == 'identity' else ftype.capitalize()+int(fcustom != [1, 1, 1, 1, 0])*f': {fcustom}'+int(ftype == 'greyscale')*f': {fgreycol} colors'+'\n'
        text += f'Contrast: {fcon}'*int(fcon != 1) + ('|'*int(fcon != 1) + f'Saturation: {fsat}')*int(fsat != 1) + ('|'*int(fcon != 1 or fsat != 1) + f'Luminance: {flum}')*int(flum != 1)+'\n'
        text += f'Pixelization: {fpix}\n' if fpix > 1 else ''
        text += f'Palette: {fpalette}\n' if isinstance(fpalette, str) else ''
        text += '' if not fdittype else f'Dithering: {fdittype.capitalize()}'+int(ftype == "change_palette")*f', {len(im.PALETTES[fpalette if isinstance(fpalette, str) else "red"])} colors\n'+int(ftype != "change_palette" and fdittype != "halftone")*f', {fditcol} colors\n'
        text += '' if not fmirror else f'Mirror: {fmirror.capitalize()}\n'
        self.show_action_label(text)

        return 1

    def add_noise(self, args=[]):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        size = get_image_size(self.cached_paths[self.current_action])
        if not args:
            self.in_other_window = True
            dialog = NoiseWindow(self.root, size, self.theme)
            self.in_other_window = False
            ntype = dialog.noise_type
            nstr = dialog.strength
            ntra = dialog.transparent_check
            ninv = dialog.invert_check
        else:
            ntype = args[0]
            nstr = args[1]
            ntra = bool(args[2])
            ninv = bool(args[3])
        if not all((ntype, nstr)): return None
        nstr = float(nstr)
        nim, new_images = [], np.zeros(len(self.frames)).tolist()
        for _ in self.frames:
            nim.append(im.noise(size, ntype, transparent=ntra, invert=ninv))
        for i in range(len(self.frames)):
            new_images[i] = Image.fromarray(im.imageInterp(np.array(self.frames[i]), nim[i], interp1=1, interp2=nstr), "RGBA")
        self.frames = copy.deepcopy(new_images)
        self.action_list.append([self.frames, self.file_path])
        if len(self.action_list) > self.action_size:
            self.action_list.pop(0)
        self.current_action = len(self.action_list) - 1
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(folder_path, exist_ok=True)
        self.im_num += 1
        self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
        if not self.file_path.endswith(('.psd', '.cur')):
            self.save_image(self.cached_file_path)
            if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
        self.update_frame()
        text = "Noise added:\n"
        text += f'Noise type: {ntype}\n'
        text += f'Strength: {nstr}\n'
        text += f'Transparent: {ntra}\n'
        text += f'Invert: {ninv}\n'
        self.show_action_label(text)

    def convert_ascii(self, event=None):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        self.in_other_window = True
        window = AsciiWindow(self.root, self.theme, self.frames)
        self.in_other_window = False

    def update_displayed_layer(self, event):
        selected_index = self.combo.current()
        if selected_index is not None:
            self.frames[0] = self.psd_layers[selected_index].compose() if selected_index < len(self.combo['values'])-1 else self.psd_layers.composite()
            self.update_frame()

    def draw_checkered_pattern(self, checker_color):
        N = 40
        if self.transparent:
            for row in range(0, self.canvas.winfo_height(), N):
                for col in range(0, self.canvas.winfo_width(), N):
                    color = checker_color[0] if (row // N + col // N) % 2 == 0 else checker_color[1]
                    self.canvas.create_rectangle(col, row, col + N, row + N, fill=color, outline="")


    def show_fps(self):
        if self.fps:
            self.fps_label = ttk.Label(self.root, text="", background=self.button_color, foreground=self.text_color,
                                       font=("Verdana", 12))
            self.fps_label.place(x=10, y=self.canvas.winfo_height())

            self.update_fps()
            print("\033[32mFPS enabled.\033[0m")
            self.display_output("FPS enabled.")
        else:
            if self.fps_label: self.fps_label.destroy()
            self.fps_label = None
            print("\033[32mFPS disabled.\033[0m")
            self.display_output("FPS disabled.")
        self.frame = self.update_frame()

    def enable_fps(self, state=None):
        if not state: self.fps = not self.fps
        else: self.fps = True if state == "enable" else False
        self.show_fps()

    def enable_checker(self):
        self.transparent = not self.transparent
        self.update_frame()

    def open_options_window(self, event):
        options_window = tk.Toplevel(self.root)
        options_window.title('Developer tools')

        options_window.resizable(False, False)

        window_width = 500
        window_height = 300
        screen_width = options_window.winfo_screenwidth()
        screen_height = options_window.winfo_screenheight()
        x_coordinate = int((screen_width - window_width) / 2)
        y_coordinate = int((screen_height - window_height) / 2)
        options_window.geometry(f'{window_width}x{window_height}+{x_coordinate}+{y_coordinate}')

        font_size = 14
        font = ('Courier', font_size)

        checkboxes = []
        for option, value in self.options.items():
            checkbox_var = tk.BooleanVar(value=value)
            checkbox = tk.Checkbutton(options_window, text=option, variable=checkbox_var, font=font)
            checkbox.pack(anchor=tk.W)
            checkboxes.append((option, checkbox_var))

        initial_checkbox_states = {option: checkbox_var.get() for option, checkbox_var in checkboxes}

        apply_button = tk.Button(options_window, text='Apply', command=lambda: self.apply_options(checkboxes, options_window, initial_checkbox_states),
                                 font=font)
        apply_button.pack(pady=10, side=tk.BOTTOM)

        options_window.grid_rowconfigure(1, weight=1)

    def open_cmd(self, event):
        if self.cmd:
            self.cmd.destroy()
            self.current = 0
            self.cmd_input = None
            self.cmd_output = None
            self.cmd = None
            self.options['Activate Console'] = True # this was set to False (in case something breaks)
            return None
        self.options['Activate Console'] = False
        self.window_cmd()

    def apply_options(self, checkboxes, options_window, initial_checkbox_states):
        errs = []
        if options_window == None or initial_checkbox_states == None:
            self.window_cmd() if self.options['Activate Console'] else None
            return None
        for option, checkbox_var in checkboxes:
            self.options[option] = checkbox_var.get()
            try:
                if option == 'Activate Console' and self.options[option] != initial_checkbox_states[option]:
                    if self.options[option]:
                        print("\033[32mConsole activated, opening...\033[0m")
                        self.window_cmd()
                    else:
                        print("\033[32mConsole deactivated, closing...\033[0m")
                        self.cmd.destroy()
                if option == 'Use Optimizations':
                    if self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.optimizations = True
                        self.display_output("Optimizations enabled.")
                        print("\033[32mOptimizations enabled.\033[0m")
                    elif not self.options[option]:
                        self.optimizations = False
                        self.display_output("Optimizations disabled.")
                        print("\033[32mOptimizations disabled.\033[0m")
                if option == 'Show FPS':
                    if self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.enable_fps("enable")
                    elif not self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.enable_fps("disable")
                if option == 'Smooth Zoom':
                    if self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.smooth_zoom = True
                    elif not self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.smooth_zoom = False
                if option == 'Copy Path':
                    if self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.copy_image = False
                    elif not self.options[option] and self.options[option] != initial_checkbox_states[option]:
                        self.copy_image = True
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", f"Couldn't apply option: '{option}'")
                errs.append(option)
        if not errs: pass
        else:
            plural = "s" if len(errs) > 1 else ''
            formatted_string = f"Option{plural} " + "{} couldn't be applied.".format(', '.join(f"'{element}'" for element in errs))
            print(f"\033[31m{formatted_string}\033[0m")
            self.show_action_label(f"Option{plural} " + "{} couldn't be applied.".format(', '.join(f"'{element}'" for element in errs)))

        options_window.destroy()
        self.update_frame()

    def window_cmd(self):
        self.cmd = tk.Toplevel(self.root)
        window_width = 1100
        window_height = 600
        screen_width = self.cmd.winfo_screenwidth()
        screen_height = self.cmd.winfo_screenheight()
        x_coordinate = int((screen_width - window_width) / 2)
        y_coordinate = int((screen_height - window_height) / 2)
        self.cmd.geometry(f'{window_width}x{window_height}+{x_coordinate}+{y_coordinate}')
        self.cmd.title("Command Prompt")
        self.cmd.configure(bg=self.cmd_color)
        self.cmd.protocol("WM_DELETE_WINDOW", self.cmd_closed)

        def get_input():
            input = self.entry_var.get()
            print(f"\033[92mCommand: {input}\033[0m")
            self.command_inputs.append(input)
            self.command_parser(input)
            self.entry_var.set('')
            self.current = 0

        self.cmd_output = tk.Text(self.cmd, wrap=tk.WORD, height=10, font=('Courier New', 14),
                                  bg=self.cmd_color, fg=self.text_color_cmd)
        self.cmd_output.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.entry_var = tk.StringVar()
        self.cmd_input = tk.Entry(self.cmd, textvariable=self.entry_var, font=('Courier New', 14), insertbackground=self.text_color_cmd, bg=self.prompt_color, fg=self.text_color_cmd)
        self.cmd_input.pack(side=tk.LEFT, ipady=6, padx=10, pady=10, fill=tk.X, expand=True, anchor='s')
        self.cmd_input.focus_set()

        cmd_button = tk.Button(self.cmd, text="Enter", command=get_input, font=('Courier New', 14), bg=self.button_color, fg=self.text_color)
        cmd_button.pack(side=tk.LEFT, padx=10, pady=10, anchor='s')
        self.buttons["cmd_button"] = cmd_button

        self.cmd.bind("<F12>", lambda event: self.open_cmd(event))
        self.cmd_input.bind('<Up>', lambda event: self.on_arrow("up"))
        self.cmd_input.bind('<Down>', lambda event: self.on_arrow("down"))
        self.cmd_input.bind('<Return>', lambda event: get_input())

    def cmd_closed(self):
        self.options['Activate Console'] = False
        self.cmd.destroy()
        self.cmd = None
        self.cmd_input = None
        self.cmd_output = None

    def display_output(self, output):
        if self.cmd_output:
            self.cmd_output.insert(tk.END, output + '\n')
            self.cmd_output.see(tk.END)

    def on_arrow(self, arrow):
        if not self.command_inputs: return None
        if arrow == "up":
            self.current += 1
        else:
            self.current -= 1
        if self.current >= len(self.command_inputs): self.current = len(self.command_inputs)
        if self.current < 1:
            self.current = 0
            self.entry_var.set('')
            return None
        self.entry_var.set(self.command_inputs[-self.current])
        self.cmd_input.icursor(len(self.cmd_input.get()))

    def command_parser(self, input):
        match = re.match(r'(\S+)\s*(.*)', input)
        if not match: return None
        command = match.group(1)
        args = match.group(2)
        if command not in COMMAND_LIST:
            self.display_output(f"`{command}` is not recognized. Type `-help` for a list of commands. \nCommands are of the form `-command --arg1 value1 --arg2 value2 ...`")
            return None
        if command == '-help': self.display_output("List of commands (`-command, (usage/info)`):\n" + "\n".join([f"   {command}{usage}" for command, usage in zip(COMMAND_LIST_DISPLAY, USAGE_LIST)]))
        if command == '-filter':
            pattern = re.compile(r'(--\S+)\s*(.*)')
            args = pattern.match(args)
            if not args:
                self.display_output("No arguments provided or invalid usage.\nUsage: `-filter --filter_type --arg1 value1 --arg2 value2 ...`")
                return None
            filter_type = args.group(1)
            args_parsed = args.group(0).split(" ", 1)
            if len(args_parsed) <= 1:
                args_parsed = ''
            else:
                args_parsed = args_parsed[1]
            filter_type = filter_type.replace('--', '')
            args_parsed = args_parsed.replace(' --', ',')
            args_parsed = args_parsed[2:].replace(' ', '=')
            img = self.apply_filter([filter_type, args_parsed], do_it=True)
            if not img:
                txt = f"Couldn't apply `{filter_type}` filter"
                self.display_output(txt + "." if not args_parsed else txt + f" with arguments `{args_parsed}`.")
                return None
            txt = f"Successfully applied `{filter_type}` filter"
            self.display_output(txt+"." if not args_parsed else txt+f" with arguments `{args_parsed}`.")
        if command == '-kernel':
            pattern = re.compile(r'(--\S+)\s*(.*)')
            args = pattern.match(args)
            if not args:
                self.display_output("No arguments provided or invalid usage.\nUsage: `-kernel --kernel_type --arg1 value1 --arg2 value2 ...`")
                return None
            kernel_type = args.group(1)
            args_parsed = args.group(0).replace(kernel_type, '')
            kernel_type = kernel_type.replace('--', '')
            args_parsed = args_parsed.replace(' --', ',')
            args_parsed = args_parsed[1:].replace(' ', '=')
            img = self.apply_kernel([kernel_type, args_parsed])
            if not img:
                txt = f"Couldn't apply `{kernel_type}` kernel"
                self.display_output(txt + "." if not args_parsed else txt + f" with arguments `{args_parsed}`.")
                return None
            txt = f"Successfully applied `{kernel_type}` kernel"
            self.display_output(txt + "." if not args_parsed else txt + f" with arguments `{args_parsed}`.")
        if command == '-load':
            if not args or not isinstance(args, str):
                self.display_output(
                    "No arguments provided or invalid usage.\nUsage: `-load --file_path`")
                return None
            args = args.replace('--', '')
            img = self.change_image(args)
            if not img:
                self.display_output(f"Couldn't load `{args}`.")
                return None
            self.display_output(f"Successfully loaded `{args}`.")
            self.entry_var.set('')
        if command == '-optimize':
            if self.optimizations:
                self.optimizations = False
                self.display_output("Optimizations disabled.")
            else:
                self.optimizations = True
                self.display_output("Optimizations enabled.")
            self.options["Use Optimizations"] = self.optimizations
            self.update_frame()
        if command == '-clc': self.cmd_output.delete("1.0", tk.END)
        if command == '-clear':
            self.new_image()
        if command == '-current': self.display_output(f"Current file loaded: `{self.file_path}`")
        if command == '-resize':
            args = args.replace(" ", "")
            args_parsed = args[2:].split('--')
            self.resize_image(args_parsed)
        if command == '-fps':
            self.enable_fps()
            self.options["Show FPS"] = self.fps
        if command == '-save':
            if not args:
                self.display_output("No arguments provided or invalid usage.\nUsage: `-save --save_path`")
                return None
            try:
                self.save_image(save_p=args.replace("--", "").replace("save_path ", ""))
            except Exception as e:
                traceback.print_exc()
                self.display_output(f"Error: {e}")
                return None
            self.display_output(f"Successfully saved image to `{os.path.abspath(args) if not os.path.isabs(args) else args}`.")
        if command == '-undo':
            args = args.replace("--", "").replace("set_limit ", "")
            self.undo(args)
        if command == '-redo':
            self.redo(args)
        if command == '-speed':
            args = args.lower().replace('--','').replace('speed ', '').replace('type', '')
            args = ' '.join(args.split()).split(' ')
            self.change_speed(args)
        if command in ['-exit', '-close', '-quit']:
            self.cmd.destroy()
            self.cmd = None
            self.cmd_input = None

    def quality(self, qlt=None):
        if not qlt: self.resize_method = Image.Resampling.BICUBIC if self.resize_method == Image.Resampling.NEAREST else Image.Resampling.NEAREST
        else: self.resize_method = qlt
        self.update_frame()

    def change_theme(self, chng=None):
        if not chng: self.theme = "white" if self.theme == "dark" else "dark"
        else:
            self.theme = chng
        if self.theme == "dark":
            self.bg_color = "#444444"
            self.button_color = "#343434"
            self.canvas_color = "#676767"
            self.text_color = "#FFFFFF"
            self.checker_color = "#898989", "#676767"
            self.cmd_color = "#121212"
            self.prompt_color = "#121212"
            self.text_color_cmd = "#FFFFFF"
        else:
            self.bg_color = "#ABABAB"
            self.button_color = "#EEEEEE"
            self.canvas_color = "#BCBCBC"
            self.text_color = "black"
            self.checker_color = "#EFEFEF", "#BCBCBC"
            self.cmd_color = "#DDDDDD"
            self.prompt_color = "#BBBBBB"
            self.text_color_cmd = "#000000"
        for button in self.buttons:
            if (button == "cmd_button" and not self.cmd) or ():
                pass
            else:
                self.buttons[button].config(bg=self.button_color, fg=self.text_color,
                                            activebackground=self.button_color, activeforeground=self.text_color,
                                            font=('Verdana', 10), bd=0, cursor="hand2", highlightthickness=3)
        for menu in self.menus:
            self.menus[menu].config(bg=self.button_color, fg=self.text_color,
                                   activeforeground=self.text_color, font=('Verdana', 10), borderwidth=0)
        self.button_frame.configure(bg=self.bg_color)
        self.root.configure(bg=self.bg_color)
        if self.cmd: self.cmd.configure(bg=self.cmd_color)
        if self.cmd_input: self.cmd_input.config(insertbackground=self.text_color_cmd, bg=self.prompt_color, fg=self.text_color_cmd)
        if self.cmd_output: self.cmd_output.config(bg=self.cmd_color, fg=self.text_color_cmd)
        self.canvas.configure(bg=self.canvas_color)
        if self.fps: self.fps_label.config(background=self.button_color, foreground=self.text_color)
        self.draw_checkered_pattern(self.checker_color)
        self.info_label.configure(bg=self.bg_color, fg=self.text_color)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', background=self.button_color, fieldbackground=self.button_color,
                        foreground=self.text_color, font=('Verdana', 10), padding=(10, 5),
                        darkcolor=self.bg_color, lightcolor=self.bg_color)
        if self.outline:
            self.outline_slider.configure(bg=self.bg_color, fg=self.text_color)
            self.outline_type.configure(background=self.bg_color, foreground=self.text_color)
            self.outline_color_select.configure(background=self.button_color, foreground=self.text_color,
                                        activebackground=self.button_color, activeforeground=self.text_color,
                                        font=('Verdana', 10), bd=0, cursor="hand2", highlightthickness=3)
            self.outline_threshold_select.configure(bg=self.bg_color, fg=self.text_color)
            self.thck_label.configure(bg=self.bg_color, fg=self.text_color)
            self.type_label.configure(bg=self.bg_color, fg=self.text_color)
            self.thr_label.configure(bg=self.bg_color, fg=self.text_color)
            self.empty_label.configure(bg=self.bg_color, fg=self.text_color)
            self.empty_label2.configure(bg=self.bg_color, fg=self.text_color)
        if self.action_label:
            self.action_label.configure(bg=self.bg_color, fg=self.text_color)
        self.gif_maker_canvas.configure(bg=self.bg_color)
        self.gif_maker_frame.configure(bg=self.bg_color)
        self.changed_gif_maker = True
        self.update_frame()

    def update_frame(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.color_picker_button.place(x=canvas_width - 75, y=canvas_height - 47)
        self.color_picker_button.lift()
        self.magic_wand_button.place(x=canvas_width - 75, y=canvas_height - 120)
        self.magic_wand_button.lift()
        if not self.frames:
            self.canvas.delete("all")
            self.draw_checkered_pattern(self.checker_color)

            col = self.color_picked[1 if self.picking_color or self.pressed_escape else 0]
            hexcol = f"#{hex(im.rgbToHex(col))[2:].zfill(6)}"
            self.info_label.configure(text=f"Image info: No image loaded."
                                           f" Zoom method: {str(self.resize_method).replace('Resampling.', '')}."
                                           f" Current image: {self.current_action + 1}/{len(self.action_list)}."
                                           f" Color picked: {col} | {hexcol}",
                                      bg=self.bg_color, fg=self.text_color)
            self.info_label.pack(side=tk.BOTTOM)
            self.canvas.create_rectangle(canvas_width - 100, canvas_height - 75, canvas_width - 76, canvas_height - 10, outline="black", fill=hexcol)

            return None

        frame = self.frames[self.current_frame]
        if frame:
            if 0 <= frame.width*self.zoom_factor*1.2 <= canvas_width or \
                    0 <= frame.height*self.zoom_factor*1.2 <= canvas_height or not self.optimizations:
                cropped_frame = frame.resize(
                    (int(frame.width * self.zoom_factor), int(frame.height * self.zoom_factor)),
                    self.resize_method)
                self.photo = ImageTk.PhotoImage(cropped_frame)
                self.canvas.delete("all")
                self.draw_checkered_pattern(self.checker_color)
                self.canvas.create_image(
                    canvas_width / 2 + self.pan_offset_x,
                    canvas_height / 2 + self.pan_offset_y,
                    anchor=tk.CENTER,
                    image=self.photo
                )
            else:
                crop_x1 = (-self.pan_offset_x - canvas_width / 6) / self.zoom_factor
                crop_y1 = (-self.pan_offset_y + canvas_height / 2) / self.zoom_factor
                crop_x2 = (canvas_width - self.pan_offset_x - canvas_width / 6) / self.zoom_factor + 2
                crop_y2 = (canvas_height - self.pan_offset_y + canvas_height / 2) / self.zoom_factor + 2

                crop_tuple = (crop_x1, crop_y1, crop_x2, crop_y2)
                cropped_frame = frame.convert("RGBA").crop(crop_tuple)

                cropped_frame = cropped_frame.resize(
                    (int(cropped_frame.width * self.zoom_factor), int(cropped_frame.height * self.zoom_factor)),
                    self.resize_method)
                self.photo = ImageTk.PhotoImage(cropped_frame)
                self.canvas.delete("all")
                self.draw_checkered_pattern(self.checker_color)
                self.canvas.create_image(
                    canvas_width / 2,
                    canvas_height / 2,
                    anchor=tk.CENTER,
                    image=self.photo
                )

            if self.file_path:
                size = self.frames[0].size
                _, file_extension = os.path.splitext(self.action_list[self.current_action][1])
            else:
                size = (0, 0)
                file_extension = ''
            col = self.color_picked[1 if self.picking_color or self.pressed_escape else 0]
            hexcol = f"#{hex(im.rgbToHex(col))[2:].zfill(6)}"
            self.info_label.configure(text=f"Image info: {size[0]}x{size[1]}, `{file_extension.lower()[1:]}`."
                                       f" Zoom method: {str(self.resize_method).replace('Resampling.', '')}."
                                           f" Current image: {self.current_action+1}/{len(self.action_list)}."
                                           f" Color picked: {col} | {hexcol}", bg=self.bg_color, fg=self.text_color)
            self.info_label.pack(side=tk.BOTTOM)
            if self.action_label: self.action_label.place(x=10, y=40)
            self.canvas.create_rectangle(canvas_width - 100, canvas_height - 75, canvas_width - 76, canvas_height - 10, outline="black", fill=hexcol)
            if self.changed_gif_maker:
                i, cnt = 0, 1
                self.gif_maker_canvas.delete("all")
                self.gif_maker_frame.pack(side=tk.BOTTOM, fill=tk.X)
                self.gif_maker_canvas.pack(side=tk.BOTTOM)
                self.gif_maker_images = list()
                if self.gif_maker_list:
                    for image in self.gif_maker_list:
                        w, h = image.size
                        w = round(w * 80 / h)
                        img = ImageTk.PhotoImage(image.resize((w, 80)))
                        self.gif_maker_images.append(img)
                        self.gif_maker_canvas.create_image(10 + i + w // 2, 50, image=self.gif_maker_images[-1])
                        self.gif_maker_canvas.create_text(10 + i + w // 2, 100, text=f"Frame {cnt}", fill=self.text_color)
                        i += w + 10
                        cnt += 1
                    self.gif_maker_canvas.config(scrollregion=(0, 0, i, 100))
                else:
                    self.gif_maker_frame.pack_forget()
                self.gif_maker_frame.update()
                self.changed_gif_maker = False


    def change_speed(self, args):
        self.speed = float(args[0])
        self.delays = self.original_delays
        if len(args) < 2 or args[1] == "relative" or args[1] == "rel":
            self.delays = list(np.clip(np.array(self.delays)/self.speed, 1, 10000).astype(np.uint16))
        elif args[1] == "absolute" or args[1] == "abs":
            self.delays = list(np.clip(np.array(self.delays)+self.speed, 1, 10000).astype(np.uint16))
        print(f"\033[94mNew delays (ms): {';'.join(str(delay) for delay in self.delays)}\033[0m")
        self.display_output(f"New delays (ms): {';'.join(str(delay) for delay in self.delays)}")
        self.update_frame()

    def update_image_position(self, event):
        self.canvas.delete("all")
        self.draw_checkered_pattern(self.checker_color)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.fps: self.fps_label.place(x=10, y=self.canvas.winfo_height())

    def start_pan(self, event):
        if self.can_pan:
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def pan(self, event):
        if self.can_pan:
            if self.enable_pan:
                delta_x = event.x - self.pan_start_x
                delta_y = event.y - self.pan_start_y
                self.pan_offset_x += delta_x
                self.pan_offset_y += delta_y
                self.pan_start_x = event.x
                self.pan_start_y = event.y

            self.update_frame()

    def stop_pan(self, event):
        self.keep_paint = False
        if self.can_pan:
            self.enable_pan = True

    def reset_view(self):
        self.current_frame = 0
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.update_frame()

    def play(self):
        if not self.action_list:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        if not is_gif(self.action_list[self.current_action][1]) or len(self.frames) <= 1:
            return 1
        self.play_button["state"] = "disabled"
        self.pause_button["state"] = "active"
        self.stop_button["state"] = "active"
        if self.animation_thread and self.animation_thread.is_alive(): return
        self.animation_thread = Thread(target=self.animate)
        self.animation_thread.start()

    def pause(self, event=None, from_frame=False):
        try:
            if not is_gif(self.action_list[self.current_action][1]) or len(self.frames) <= 1:
                return
        except:
            pass
        if not self.action_list:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return
        if self.pause_button["state"] == "disabled" and not from_frame:
            self.play()
            return
        self.play_button["state"] = "active"
        self.pause_button["state"] = "disabled"
        self.stop_button["state"] = "active"
        try: self.root.after_cancel(self.animation_id)
        except: pass
        self.update_frame()
        return True

    def prev_frame(self, event=None):
        passed = self.pause(from_frame=True)
        if not passed: return
        self.current_frame -= 1 if self.current_frame > 0 else 0
        self.update_frame()

    def next_frame(self, event=None):
        passed = self.pause(from_frame=True)
        if not passed: return
        self.current_frame = self.current_frame+1 if self.current_frame < len(self.frames)-1 else len(self.frames)-1
        self.update_frame()

    def stop(self):
        if not self.action_list:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None

        self.play_button["state"] = "active"
        self.pause_button["state"] = "disabled"
        self.stop_button["state"] = "disabled"
        self.current_frame = 0
        self.update_frame()
        if not is_gif(self.action_list[self.current_action][1]) or len(self.frames) <= 1:
            return 1
        try: self.root.after_cancel(self.animation_id)
        except: pass
        self.update_frame()

    def animate(self):
        if len(self.frames) <= 1: return None
        ldel = len(self.delays)
        if ldel <= 1 or ldel != len(self.frames):
            tmp = self.file_path
            self.file_path = self.action_list[self.current_action][1]
            _, self.delays = self.load_frames()
            self.file_path = tmp
        self.current_frame += 1
        if self.current_frame == len(self.frames):
            self.current_frame = 0
        delay = int(self.delays[self.current_frame])
        self.animation_id = self.root.after(delay, self.animate)
        self.update_frame()

    def on_mousewheel(self, event):
        if not self.in_other_window and self.file_path:
            x, y = self.canvas.canvasx(event.x) - self.canvas.winfo_width() / 2, self.canvas.canvasy(
                event.y) - self.canvas.winfo_height() / 2
            if event.delta > 0:
                self.zoom_in(x=x, y=y)
            else:
                self.zoom_out(x=x, y=y)

    def zoom_in(self, event=None, x=None, y=None):
        if not x and not y:
            x, y = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx() - self.canvas.winfo_width() // 2, \
                   self.canvas.winfo_pointery() - self.canvas.winfo_rooty() - self.canvas.winfo_height() // 2
        if not self.frames: return
        size = self.frames[0].width, self.frames[0].height
        if self.zoom_factor*max(size[0], size[1]) < 4*(max(size[0], size[1])**2)/self.zoom_factor:
            self.zoom_update(1.2, x, y)

    def zoom_out(self, event=None, x=None, y=None):
        if not x and not y:
            x, y = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx() - self.canvas.winfo_width() // 2, \
                   self.canvas.winfo_pointery() - self.canvas.winfo_rooty() - self.canvas.winfo_height() // 2
        if not self.frames: return
        if self.zoom_factor*max(self.frames[0].width, self.frames[0].height) > 100:
            self.zoom_update(1 / 1.2, x, y)

    def zoom_update(self, factor, x, y):
        if self.smooth_zoom:
            if not self.zoom_animation_in_progress:
                self.zoom_animation_in_progress = True
                for i in range(8):
                    t = (i+1)/8
                    incremental_factor  = 1+(factor-1)*(t*t*(1.5-1*t))
                    self.zoom_factor *= incremental_factor

                    original_distance_x = x - self.pan_offset_x
                    original_distance_y = y - self.pan_offset_y

                    self.pan_offset_x = x - original_distance_x * incremental_factor
                    self.pan_offset_y = y - original_distance_y * incremental_factor

                    self.update_frame()
                    self.root.update()
                self.zoom_animation_in_progress = False
        else:
            self.zoom_factor *= factor

            original_distance_x = x - self.pan_offset_x
            original_distance_y = y - self.pan_offset_y

            self.pan_offset_x = x - original_distance_x * factor
            self.pan_offset_y = y - original_distance_y * factor

            self.update_frame()
        self.update_frame()

    def on_window_resize(self, event):
        self.update_frame()

    def change_path(self):
        new_paths = filedialog.askopenfilename(filetypes=[("Image, gif and video files", FILE_SUPPORTED+";"+VIDEO_SUPPORTED)],
                                              multiple=True)
        if not new_paths: return
        for path in new_paths:
            self.change_image(path)

    def change_image(self, path):
        is_video_path = is_video(path)
        if is_video_path:
            try:
                self.convert_video_to_gif(path)
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", "An error occurred when converting the video to GIF, it might be too large.")
        elif not is_image(path):
            messagebox.showerror("Invalid file type", "Please select a valid image file.")
            return None
        self.file_path = "output.gif" if is_video_path else path
        if self.animation_thread:
            self.stop()
        self.reset_view()
        size = get_image_size(self.file_path)
        if size[0]*size[1] > 178956970:
            ask = messagebox.askyesno("Image exceeds limit",
                                      f"Image size ({size[0]*size[1]} pixels. {size[0]}x{size[1]} resolution) exceeds PIL limit of 178956970 pixels, "
                                      f"could be decompression bomb DOS attack. \nProceed anyway?")
            if not ask: return None
        if len(self.action_list) > self.action_size:
            self.action_list.pop(0)
        self.current_action = len(self.action_list)
        self.original_frames, self.original_delays = self.load_frames()
        self.delays = self.original_delays.copy()
        self.frames = self.original_frames.copy()
        self.current_frame = 0
        self.apply_filter(['identity', ''], do_it=False)
        self.change_theme(chng=self.theme)
        if self.outline:
            self.outline_slider.set(1)
            self.outline_slider.config(to=int(min(self.frames[0].width, self.frames[0].height) / 8))
        if not is_gif(self.file_path): self.stop()
        if is_video_path:
            try: os.remove('output.gif')
            except: pass
        self.play()
        return 1

    def save_image(self, save_p=None):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        if len(self.frames) > 1:
            save_path = filedialog.asksaveasfilename(defaultextension=".gif",
                                                     filetypes=[("GIF and Image files", SAVE_SUPPORTED)]) if not save_p else save_p
            if not save_path:
                return None

            _, file_extension = os.path.splitext(save_path)
            if file_extension.lower() not in SAVE_FILETYPES:
                messagebox.showerror("Unsupported file type", f"Saving to {file_extension.upper()[1:]} is not supported.")
                return None

            if file_extension != ".gif":
                if file_extension in [".jpg", ".jpeg", ".jfif"]:
                    self.frames[self.current_frame] = self.frames[self.current_frame].convert("RGB")
                self.frames[self.current_frame].save(save_path)
                return 0

            # to save gifs with transparency
            gif_frames = []
            for i, frame in enumerate(self.frames):
                frame = frame.convert("RGBA")
                alpha = frame.split()[3]
                new_frame = Image.new("RGBA", frame.size, (0, 0, 0, 0))
                new_frame.paste(frame, (0, 0), mask=alpha)
                gif_frames.append(new_frame)

            gif_frames[0].save(save_path, save_all=True, append_images=gif_frames[1:], duration=self.delays, loop=0,
                               disposal=2)
        else:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("GIF and Image files", SAVE_SUPPORTED)]) if not save_p else save_p
            if not save_path:
                return None

            _, file_extension = os.path.splitext(save_path)
            if file_extension.lower() not in SAVE_FILETYPES:
                messagebox.showerror("Unsupported file type", f"Saving to {file_extension.upper()[1:]} is not supported.")
                return None

            img = self.frames[0]
            if save_path.lower().endswith((".jpg", ".jpeg", ".jfif")):
                img = img.convert("RGB")
            img.save(save_path)
        if not save_p:
            self.show_action_label(f"Saved at {save_path}.")

    def filter_text(self):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None

        texts = ["Filter:", "Parameters:"]

        user_inputs = []
        for text in texts:
            user_input = simpledialog.askstring("", text)
            if user_input is None:
                break
            user_inputs.append(user_input)
        if user_inputs:
            print(f"Applying filter `{user_inputs[0]}` with parameters: `{'Default' if not user_inputs[1] else user_inputs[1]}`")
            self.apply_filter(inp=user_inputs, do_it=True)
            print("Applied.")

    def kernel_text(self):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        texts = ["Kernel:", "Parameters:"]

        user_inputs = []
        for text in texts:
            user_input = simpledialog.askstring("", text)
            if user_input is None:
                break
            user_inputs.append(user_input)
        if user_inputs:
            print(f"Applying kernel `{user_inputs[0]}` with parameters: `{'Default' if not user_inputs[1] else user_inputs[1]}`")
            self.apply_kernel(inp=user_inputs)
            print("Applied.")

    def apply_filter(self, inp, do_it):
        filter = inp[0]
        inp[1] = inp[1].replace(" ", "").replace("{", "[").replace("(", "[").replace("}", "]").replace(")", "]")
        a = coma_split(inp[1])
        params = [m for m in a]
        try:
            if params[0] == '':
                self.frames = [Image.fromarray(im.imageFilter(np.array(frame.convert("RGBA")), filter)) for frame in self.original_frames]
            else:
                param_dict = {param.split('=')[0]: check_type(param.split('=')[1]) for param in params}
                self.frames = [Image.fromarray(im.imageFilter(np.array(frame.convert("RGBA")), filter, **param_dict)) for frame in self.original_frames]
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            traceback.print_exc()
            return None
        self.current_frame = 0

        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(folder_path, exist_ok=True)
        self.im_num += 1
        self.cached_file_path = folder_path+f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
        if not self.file_path.endswith(('.psd', '.cur')):
            self.save_image(self.cached_file_path)
            if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
        self.action_list.append([self.frames, self.file_path])
        if do_it:
            if len(self.action_list) > self.action_size:
                self.action_list.pop(0)
            self.current_action = len(self.action_list) - 1
        self.update_frame()

        return 1

    def apply_kernel(self, inp):
        kernel = inp[0]
        inp[1] = inp[1].replace(" ", "").replace("{","[").replace("(","[").replace("}","]").replace(")","]")
        a = coma_split(inp[1])
        params = [m for m in a]
        try:
            if params[0] == '':
                self.frames = [Image.fromarray(im.imageKernel(np.array(frame.convert("RGBA")), kernel)) for frame in self.original_frames]
            else:
                param_dict = {param.split('=')[0]: check_type(param.split('=')[1]) for param in params}
                self.frames = [Image.fromarray(im.imageKernel(np.array(frame.convert("RGBA")), kernel, **param_dict)) for frame in self.original_frames]
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            traceback.print_exc()
            return None

        self.current_frame = 0

        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(folder_path, exist_ok=True)
        self.im_num += 1
        self.cached_file_path = folder_path + f"\\cached_image_{self.im_num}{os.path.splitext(self.file_path)[1]}"
        if not self.file_path.endswith(('.psd', '.cur')):
            self.save_image(self.cached_file_path)
            if not self.cached_file_path in self.cached_paths: self.cached_paths.append(self.cached_file_path)
        self.action_list.append([self.frames, self.file_path])
        if len(self.action_list) > self.action_size:
            self.action_list.pop(0)
        self.current_action = len(self.action_list) - 1

        self.update_frame()
        return 1

    def apply_changes(self):
        if not self.file_path:
            messagebox.showerror("File not loaded", "No file loaded, please load image/gif/video.")
            return None
        if messagebox.askquestion("Apply changes", "Do you want to apply changes?") == "yes":
            self.original_frames = copy.deepcopy(self.frames)
            self.original_delays = copy.deepcopy(self.delays)
            self.action_list.append([self.original_frames, self.file_path])
            if len(self.action_list) > self.action_size:
                self.action_list.pop(0)
            self.current_action = len(self.action_list) - 1
            self.update_frame()
        else: return None

    def delete_cache(self, all=False):
        try:
            print("\033[94mDeleting cached files...\033[0m")
            if all:
                folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
                os.makedirs(folder_path, exist_ok=True)
                files = os.listdir(folder_path)
                for file in files:
                    os.remove(os.path.join(folder_path, file))
            else:
                for cache in self.cached_paths:
                    os.remove(cache)
            self.cached_paths.clear()
            self.action_list.clear()
            self.current_action = -1
            self.im_num = 0
            self.new_image()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred when deleting cached files: {str(e)}")
            print("\033[91mCouldn't delete cached files.\033[0m")
            self.show_action_label("Couldn't delete cached files.")
            self.new_image()
            return None
        print("\033[92mSuccessfully deleted cached files.\033[0m")
        self.show_action_label("Successfully deleted cached files.")

    def convert_video_to_gif(self, video_path):
        print("Converting video to GIF (it may take a while)...")
        self.display_output("Converting video to GIF (it may take a while)...")

        vid = iio.get_reader(video_path, 'ffmpeg')

        fps = vid.get_meta_data()['fps']
        gif_writer = iio.get_writer('output.gif', fps=fps)

        try:
            for frame in vid:
                gif_writer.append_data(frame)
        finally:
            gif_writer.close()

        print("Successfully converted.")
        self.display_output("Successfully converted.")
        self.show_action_label("Successfully converted video to GIF.")

    def handle_drop(self, event):
        files = event.data
        if files:
            if isinstance(files, list): file_path = files[0]
            else:
                file_path = files.replace("{", "").replace("}", "")

            self.change_image(file_path)
            self.on_drag_leave(event)
            self.update_frame()

    def on_drag_enter(self, event):
        if not self.drop_rectangle:
            if self.file_path: self.pause()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.drop_rectangle = 1
            self.op_image = ImageTk.PhotoImage(Image.new('RGBA', (canvas_width+20, canvas_height+20), (0, 0, 0, 127)))
            self.canvas.create_image(canvas_width/2-10, canvas_height/2-10, image=self.op_image)
            self.canvas.create_image(canvas_width/2, canvas_height/2, image=self.dnd_image)

    def on_drag_leave(self, event):
        if self.drop_rectangle:
            self.drop_rectangle = None
            if self.file_path and len(self.frames) > 1: self.play()
            self.update_frame()


if __name__ == "__main__":
    if not os.path.exists("PARAMETERS.txt"):
        with open('PARAMETERS.txt', 'w') as f:
            f.write("fullscreen = False\ntheme = 'dark'\nopen_console = False\ntransparentbg = True\nunit = 'px'\nDPI = 96\naction_size = 32\n")
    with open("PARAMETERS.txt", 'r') as f:
        lines = f.readlines()
    parameters = {}
    for line in lines:
        if line == "\n": continue
        l = line[:line.find("#")].split("=")
        name, value = l[0].strip(), l[1].strip()
        parameters[str(name)] = ast.literal_eval(value)
    ImageEditor(**parameters)