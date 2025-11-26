#!/usr/bin/env python3
import sys
from PIL import Image

def convert_non_black_to_white(input_path: str, output_path: str):
    """
    Load an image, convert all non-black pixels to white, and save result.
    A pixel is considered black if R=G=B=0.
    """

    img = Image.open(input_path).convert("RGB")
    pixels = img.load()

    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            # If not black → turn white
            if (r, g, b) != (0, 0, 0):
                pixels[x, y] = (255, 255, 255)

    img.save(output_path)
    print(f"Saved cleaned image to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_non_black_to_white.py <input_image> <output_image>")
        sys.exit(1)

    convert_non_black_to_white(sys.argv[1], sys.argv[2])
