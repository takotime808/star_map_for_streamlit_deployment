#!/usr/bin/env python3
"""
Robust black/midtone → STL converter.
Features:
- invert flag
- midtone half-height extrusion flag
- degenerate contour removal
- hole-preserving polygon extraction
- optional morphological close
- manifold-safe STL extrusion
- debug mask dumps
"""

import argparse
import numpy as np
from PIL import Image
import cv2
from shapely.geometry import Polygon, LinearRing
import trimesh


# -------------------------------------------------------------
# POLYGON EXTRACTION UTIL
# -------------------------------------------------------------
def mask_to_polygons(mask, pixel_size_mm, min_points=4):
    """
    Convert a binary mask to shapely polygons,
    preserving holes using OpenCV's CCOMP hierarchy.
    Drops degenerate contours.
    """
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

    # ---------------------------
    # FIRST PASS: SHELLS
    # ---------------------------
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

    # ---------------------------
    # SECOND PASS: HOLES
    # ---------------------------
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
# MAIN CONVERSION FUNCTION
# -------------------------------------------------------------
def image_to_stl_black_pixels(
    input_path: str,
    output_path: str,
    height_mm: float = 5.0,
    pixel_size_mm: float = 0.5,
    threshold: int = 128,
    invert: bool = False,
    do_close: bool = True,
    midtones_half_height: bool = False,
    debug_mask_path: str = None,
):
    """
    Load image and extrude:
    - Black pixels to full height
    - Optional midtones to half height
    """

    img = Image.open(input_path).convert("L")
    arr = np.array(img)

    # ---------------------------
    # Optional inversion
    # ---------------------------
    if invert:
        arr = 255 - arr

    # ---------------------------
    # BLACK MASK
    # Black = values < threshold
    # Invert so black=255 (white mask)
    # ---------------------------
    _, bw_black = cv2.threshold(arr, threshold, 255, cv2.THRESH_BINARY_INV)

    # Morph close to merge 1px cracks
    if do_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw_black = cv2.morphologyEx(bw_black, cv2.MORPH_CLOSE, kernel)

    # Optional midtone mask:
    # midtone = gray values between threshold and *just below* pure white
    if midtones_half_height:
        # Everything from threshold+1 to 254 is midtone
        midtone_mask = np.zeros_like(arr, dtype=np.uint8)
        midtone_mask[(arr > threshold) & (arr < 255)] = 255

        if do_close:
            midtone_mask = cv2.morphologyEx(midtone_mask, cv2.MORPH_CLOSE, kernel)
    else:
        midtone_mask = None

    # Save debug masks
    if debug_mask_path:
        debug = {}
        debug["black.png"] = bw_black
        if midtones_half_height:
            debug["midtones.png"] = midtone_mask

        for name, m in debug.items():
            Image.fromarray(m).save(debug_mask_path + "_" + name)

        print(f"Saved debug mask(s) to {debug_mask_path}_*.png")

    # ---------------------------
    # EXTRACT POLYGONS
    # ---------------------------
    black_polys = mask_to_polygons(bw_black, pixel_size_mm)

    if not black_polys and not midtones_half_height:
        raise RuntimeError("No black regions detected.")

    midtone_polys = []
    if midtones_half_height:
        midtone_polys = mask_to_polygons(midtone_mask, pixel_size_mm)

    # ---------------------------
    # EXTRUDE
    # ---------------------------
    meshes = []

    # BLACK = full height
    for poly in black_polys:
        tm = trimesh.creation.extrude_polygon(
            poly,
            height_mm,
            triangulation_engine="manifold"
        )
        meshes.append(tm)

    # MIDTONES = half height
    if midtones_half_height:
        half = height_mm / 2.0
        for poly in midtone_polys:
            tm = trimesh.creation.extrude_polygon(
                poly,
                half,
                triangulation_engine="manifold"
            )
            meshes.append(tm)

    # MERGE + SAVE
    if not meshes:
        raise RuntimeError("No geometry produced.")

    merged = trimesh.util.concatenate(meshes)
    merged.export(output_path)
    print(f"Saved STL → {output_path}")


# -------------------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert image → STL (black pixels, optional midtones)."
    )

    parser.add_argument("input_image")
    parser.add_argument("output_stl")

    parser.add_argument("--height-mm", type=float, default=5.0)
    parser.add_argument("--pixel-size-mm", type=float, default=0.5)
    parser.add_argument("--threshold", type=int, default=128)

    parser.add_argument("--invert", action="store_true",
                        help="Invert image BEFORE thresholding.")

    parser.add_argument("--no-close", action="store_true",
                        help="Disable morphological close.")

    parser.add_argument("--midtones-half-height", action="store_true",
                        help="Extrude midtone pixels (gray but not black/white) to half height.")

    parser.add_argument("--debug-mask", type=str,
                        help="Save debug mask(s).")

    args = parser.parse_args()

    image_to_stl_black_pixels(
        args.input_image,
        args.output_stl,
        height_mm=args.height_mm,
        pixel_size_mm=args.pixel_size_mm,
        threshold=args.threshold,
        invert=args.invert,
        do_close=not args.no_close,
        midtones_half_height=args.midtones_half_height,
        debug_mask_path=args.debug_mask,
    )
