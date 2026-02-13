import cv2
import os
import numpy as np
import random as rng
def save(out_dir, name, img):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{name}.png"))
    ok = cv2.imwrite(out_path, img)
    print("Saved:", out_path, "ok =", ok)

def process(path, out_dir):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot find the image: {path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("original shape:", image.shape, "gray shape:", gray.shape)
    save_debug(out_dir, "gray.png", gray)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    save_debug(out_dir, "hsv.png", hsv)

    lower = (0, 0, 160)
    upper = (180, 70, 255)
    mask = cv2.inRange(hsv, lower, upper)
    print(type(mask), mask.dtype, mask.shape)
    print("unique values:", np.unique(mask))

    kernel_close = np.ones((7,7), np.uint8)
    kernel_open = np.ones((5,5), np.uint8)
    close1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    save_debug(out_dir, "close1.png", close1)

    close2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    save_debug(out_dir, "close2.png", close2)

    open_after = cv2.morphologyEx(close2, cv2.MORPH_OPEN, kernel_open, iterations=1)
    save_debug(out_dir, "05_open_after_close.png", open_after)

def save_step(out_dir, name, img):
    os.makedirs(out_dir, exist_ok=True)

    # If mask is boolean, convert to 0/255
    if img.dtype == np.bool_:
        img = (img.astype(np.uint8) * 255)

    # If float 0..1, scale to 0..255
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), img)

def overlay_mask(bgr, mask, alpha=0.45):
    # mask: 0/255 single-channel
    out = bgr.copy()
    red = np.zeros_like(bgr)
    red[:, :, 2] = 255
    m = mask > 0
    out[m] = (alpha * red[m] + (1 - alpha) * bgr[m]).astype(np.uint8)
    return out

def hsv_white_mask(img_bgr, S_max=25, V_min=150):
    """
    White-ish pixels: low saturation + high value.
    OpenCV HSV ranges: H:0-179, S:0-255, V:0-255
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = (0,   0,   V_min)
    upper = (179, S_max, 255)
    white = cv2.inRange(hsv, lower, upper)
    return hsv, white

def wing_dark_mask(img_bgr, dark_max=80):
    """
    Dark wing pixels in grayscale (helps restrict search region).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, dark_max)
    return gray, dark

def clean_mask(mask, open_k=7, close_k=7):
    """
    OPEN removes thin streaks; CLOSE fills small holes.
    """
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close, iterations=1)
    return m

def build_candidate(img_bgr,
                    S_max=70, V_min=160,
                    dark_max=80, dilate_k=25,
                    open_k=7, close_k=7,
                    out_dir="debug", prefix="img"):
    # --- HSV channels + white mask ---
    hsv, white = hsv_white_mask(img_bgr, S_max=S_max, V_min=V_min)
    H, S, V = cv2.split(hsv)

    # Visualize H/S/V nicely:
    # Hue is 0..179; scale to 0..255 for viewing
    H_vis = cv2.convertScaleAbs(H, alpha=(255.0 / 179.0))
    save(out_dir, f"{prefix}_00_H", H_vis)
    save(out_dir, f"{prefix}_00_S", S)
    save(out_dir, f"{prefix}_00_V", V)
    save(out_dir, f"{prefix}_01_white_mask_raw", white)
    save(out_dir, f"{prefix}_01_overlay_white_raw", overlay_mask(img_bgr, white))

    # --- Dark wing mask + dilation (restrict region to near-wing) ---
    gray, dark = wing_dark_mask(img_bgr, dark_max=dark_max)
    save(out_dir, f"{prefix}_02_gray", gray)
    save(out_dir, f"{prefix}_02_dark_mask", dark)
    save(out_dir, f"{prefix}_02_overlay_dark", overlay_mask(img_bgr, dark))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    dark_dil = cv2.dilate(dark, k, iterations=1)
    save(out_dir, f"{prefix}_03_dark_dilated", dark_dil)
    save(out_dir, f"{prefix}_03_overlay_dark_dilated", overlay_mask(img_bgr, dark_dil))

    # --- Candidate: white pixels near wing ---
    candidate = cv2.bitwise_and(white, dark_dil)
    save(out_dir, f"{prefix}_04_candidate_raw", candidate)
    save(out_dir, f"{prefix}_04_overlay_candidate_raw", overlay_mask(img_bgr, candidate))

    # --- Clean candidate (remove thin streaks / fill holes) ---
    candidate_clean = clean_mask(candidate, open_k=open_k, close_k=close_k)
    save(out_dir, f"{prefix}_05_candidate_clean", candidate_clean)
    save(out_dir, f"{prefix}_05_overlay_candidate_clean", overlay_mask(img_bgr, candidate_clean))

    return candidate_clean

if __name__ == "__main__":
    img_path = "data/raw/IMG_20260209_141642556.jpg"   # <-- change this
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit("Could not read image. Check the path.")

    # Start with these; tune after looking at debug outputs
    build_candidate(
        img,
        S_max=60, V_min=155,     # <-- main HSV knobs
        dark_max=85, dilate_k=31, # <-- near-wing restriction knobs
        open_k=9, close_k=7,     # <-- morphology knobs
        out_dir="debug",
        prefix="sample"
    )
    print("Saved debug images to ./debug/")