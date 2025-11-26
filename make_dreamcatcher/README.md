**convert all non-black pixels to white**
```bash
python convert_non_black_to_white.py no_stars.png mapped.png
```

**Make dream-catcher from black and white image:**
```bash
python image_to_stl.py mapped.png mapped.stl --pixel-size-mm 0.5 --height-mm 5 --invert
```