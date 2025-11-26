import streamlit as st
from PIL import Image
import numpy as np
import cv2
import trimesh
import io
from shapely.geometry import Polygon, LinearRing


# -------------------------------------------------------------
# UTIL: MASK → POLYGONS
# -------------------------------------------------------------
def mask_to_polygons(mask, pixel_size_mm, min_points=4):
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    polygons = []
    idx_map = {}

    # Shells
    for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
        parent = h[3]
        if len(cnt) < min_points:
            continue

        pts = cnt[:, 0, :] * pixel_size_mm
        if parent == -1:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
                idx_map[i] = len(polygons) - 1

    # Holes
    for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
        parent = h[3]
        if parent != -1:
            if len(cnt) < min_points:
                continue

            pts = cnt[:, 0, :] * pixel_size_mm
            ring = LinearRing(pts)
            if ring.is_valid and parent in idx_map:
                shell_index = idx_map[parent]
                shell = polygons[shell_index]
                new_poly = Polygon(shell.exterior.coords,
                                   list(shell.interiors) + [ring])
                polygons[shell_index] = new_poly

    return polygons


# -------------------------------------------------------------
# SCRIPT 1: NON-BLACK → WHITE
# -------------------------------------------------------------
def convert_non_black_to_white(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    pixels = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if (r, g, b) != (0, 0, 0):
                pixels[x, y] = (255, 255, 255)

    return img


# -------------------------------------------------------------
# SCRIPT 2: IMAGE → STL
# -------------------------------------------------------------
def image_to_stl(
    img: Image.Image,
    height_mm: float,
    pixel_size_mm: float,
    threshold: int,
    invert: bool,
    do_close: bool,
    midtones_half_height: bool
):
    arr = np.array(img.convert("L"))

    if invert:
        arr = 255 - arr

    _, bw_black = cv2.threshold(arr, threshold, 255, cv2.THRESH_BINARY_INV)

    if do_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw_black = cv2.morphologyEx(bw_black, cv2.MORPH_CLOSE, kernel)

    # midtone mask
    midtone_mask = None
    if midtones_half_height:
        midtone_mask = np.zeros_like(arr, dtype=np.uint8)
        midtone_mask[(arr > threshold) & (arr < 255)] = 255
        if do_close:
            midtone_mask = cv2.morphologyEx(midtone_mask, cv2.MORPH_CLOSE, kernel)

    black_polys = mask_to_polygons(bw_black, pixel_size_mm)
    midtone_polys = []
    if midtones_half_height:
        midtone_polys = mask_to_polygons(midtone_mask, pixel_size_mm)

    meshes = []
    for poly in black_polys:
        meshes.append(trimesh.creation.extrude_polygon(poly, height_mm, triangulation_engine="manifold"))

    if midtones_half_height:
        half = height_mm / 2.0
        for poly in midtone_polys:
            meshes.append(trimesh.creation.extrude_polygon(poly, half, triangulation_engine="manifold"))

    if not meshes:
        raise RuntimeError("No geometry generated")

    merged = trimesh.util.concatenate(meshes)
    return merged


# -------------------------------------------------------------
# STREAMLIT UI (SINGLE UPLOAD)
# -------------------------------------------------------------
st.set_page_config(page_title="Image → STL Generator", layout="wide")
st.title("🧱 Image → STL Generator (with Optional Black/White Cleaning)")

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Original Image", use_column_width=True)

    # SETTINGS
    st.subheader("Processing Options")

    col1, col2 = st.columns(2)
    clean_toggle = col1.checkbox("Convert Non-Black → White (Preprocess)", value=True)
    invert = col2.checkbox("Invert Image First", value=True)

    if clean_toggle:
        cleaned = convert_non_black_to_white(img)
        st.image(cleaned, caption="Cleaned (non-black → white)", use_column_width=True)

        # Allow user to download cleaned image
        buf = io.BytesIO()
        cleaned.save(buf, format="PNG")
        st.download_button(
            "⬇ Download Cleaned Image",
            buf.getvalue(),
            file_name="cleaned_image.png",
            mime="image/png"
        )
        img = cleaned  # Use cleaned image for STL generation

    # STL parameters
    st.subheader("STL Generation Settings")
    colA, colB = st.columns(2)
    height_mm = colA.number_input("Extrusion Height (mm)", 1.0, 50.0, 5.0)
    pixel_size_mm = colB.number_input("Pixel Size (mm)", 0.1, 5.0, 0.5)

    threshold = colA.slider("Threshold for Black Detection", 0, 255, 128)
    do_close = colB.checkbox("Morphological Close (Merge Cracks)", value=True)

    midtones_half_height = st.checkbox("Extrude Midtones to Half Height")

    # BUTTON TO GENERATE STL
    if st.button("Generate STL"):
        with st.spinner("Processing image + generating STL…"):
            try:
                mesh = image_to_stl(
                    img,
                    height_mm,
                    pixel_size_mm,
                    threshold,
                    invert,
                    do_close,
                    midtones_half_height
                )

                stl_bytes = mesh.export(file_type="stl")
                st.download_button(
                    "⬇ Download STL",
                    stl_bytes,
                    file_name="output.stl",
                    mime="model/stl"
                )
                st.success("STL generated successfully!")

            except Exception as e:
                st.error(f"Error: {e}")
