# Cosk Image Editor
An Image Editor made in python, for things like adding filters, dithering, convolution-like operations, etc. Not made for painting and such.

## Features
- Load images, GIFs and videos from your computer, URL, copy-paste or drag and dropping.
- Load multiple images at once.
- Apply filters, dithering, pixelization, and more.
- Apply convolution-like operations such as blurring, motion blur, edge detection, etc.
- Add an outline, add noise or resize your images.
- Eyedropper/Magic Wand tools.
- GIF Maker to easily make gifs.
- Image to ASCII.
- Custom command prompt for more advanced options.
- Customizable Undo/Redo size.

## Usage
### Basic controls
- Pan around by pressing left-click and moving your mouse.
- Zoom with the mouse wheel, or use + and -.
- Load a file: Ctrl+O. Empty the canvas: Ctrl+N. Save a file: Ctrl+S.
- If the file is a gif; Pause and play by pressing Space, move through the frames by pressing the left and right arrow keys.
### GIF Maker
- Press Shift+plus (+) to add the current image/frame to the frame list.
- Press Shift+minus (-) to delete the last frame added, Control+Shift+minus to delete the first and Alt+minus to delete all.
- Once you have the frames, create the GIF with Shift+G.

## Showcase
These GIFs where recorded in mp4 and then converted to GIF/resized with this program.
![filters_showcase](https://github.com/Coskon/coskimageeditor/assets/54825470/a83da651-c4d3-45a8-a3f2-7b4b22492be8)
![power](https://github.com/Coskon/coskimageeditor/assets/54825470/c0ca0838-8d44-4d12-9256-0f05758b5c78)

## Installation Guide
- Clone or [download](https://github.com/Coskon/coskimageeditor/archive/refs/heads/main.zip) the repository.
- Run the `run.bat` script to execute the editor.
- **(If `run.bat` didn't work)** Open a CMD on the project folder and input the following commands (windows):
    ```console
    python.exe -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    imageeditor.py
    ```
- **(Optional)** Change initialization parameters in `PARAMETERS.txt`.

## Limitations
- Loading from an URL might be buggy.
- Converting from video to GIF takes a lot of time and memory for bigger videos.
- Command prompt needs a little more polishing.
- GIF Maker could be a little more optimized, will be changed sometime.
- Lots of performance issues. (python my beloved)

## Known bugs
- Pressing some 'Cancel' buttons might still apply the changes.
- The magic wand tool is currently not getting the correct position in the image. This will be fixed.

## Added features
- C implementation for dithering, so now it is very fast (at least compared to what it was before). The rough preview in the filter section is practically real time now.

## To be added
- [ ] Proper 'Kernel' window.
- [ ] More customizable 'Add Noise' window.
- [ ] Image cropping.
- [ ] More functions running in C.
- [ ] Euclidean Distance Transform for the outline function.
