import argparse
import os
import sys
import re
from typing import Dict, List, Optional, Tuple

import pytesseract
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher

# ---------- CONFIGURATION ----------
MODEL_TEXT = "Searching for NikeAir shoes on Myntra"
IMAGE_PATH = "result.png"   # path to your image result
SIMILARITY_THRESHOLD = 80   # 0-100; below this, flag mismatch
# ----------------------------------

def preprocess_image(image_path: str, enable_preprocess: bool = True) -> Image.Image:
    """Open and optionally preprocess the image to improve OCR accuracy.

    Preprocessing uses grayscale, autocontrast, slight sharpening, and
    light thresholding to reduce noise. These are conservative operations
    intended to help typical UI screenshots.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unable to open image (unrecognized format): {image_path}") from exc

    if not enable_preprocess:
        return image

    # Convert to grayscale and autocontrast to normalize lighting
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    # Slightly boost contrast; avoids over-sharpening text
    gray = ImageEnhance.Contrast(gray).enhance(1.5)

    # Light binarization helps for crisp UI fonts
    def _threshold(pixel: int) -> int:
        return 255 if pixel > 150 else 0

    binary = gray.point(_threshold)
    return binary


def extract_text_from_image(
    image_path: str,
    *,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    enable_preprocess: bool = True,
) -> str:
    """Extract raw text from image using Tesseract OCR.

    Parameters control the OCR engine mode (OEM) and page segmentation (PSM):
    - psm 6 is good for blocks of text
    - oem 3 lets Tesseract decide the best engine
    """
    image = preprocess_image(image_path, enable_preprocess=enable_preprocess)
    config = f"--oem {oem} --psm {psm}"
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not found in PATH. "
            "Install it and try again."
        ) from exc
    return text


def normalize(text: str) -> str:
    """Normalize text for fair matching.

    Steps:
    - Insert spaces at camelCase or letter-digit boundaries
    - Lowercase
    - Replace non-alphanumeric with spaces
    - Collapse repeated whitespace
    """
    if text is None:
        return ""

    # Insert spaces between lower->Upper (camelCase) and between letters<->digits
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)

    # Normalize common separators
    text = re.sub(r"[\-_./]+", " ", text)

    # Lowercase and remove non-alnum
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Normalize and split into tokens."""
    norm = normalize(text)
    return norm.split() if norm else []


def compute_similarity_metrics(a: str, b: str) -> Dict[str, int]:
    """Compute multiple fuzzy metrics (0-100)."""
    return {
        "char_ratio": int(fuzz.ratio(a, b)),
        "partial_ratio": int(fuzz.partial_ratio(a, b)),
        "token_sort_ratio": int(fuzz.token_sort_ratio(a, b)),
        "token_set_ratio": int(fuzz.token_set_ratio(a, b)),
    }


def get_word_differences(a: str, b: str) -> List[Dict[str, str]]:
    """Find word-level differences between two normalized texts.

    Uses difflib to produce token operations like replace/insert/delete.
    """
    a_words, b_words = a.split(), b.split()
    sm = SequenceMatcher(None, a_words, b_words)
    diffs: List[Dict[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        diffs.append(
            {
                "type": tag,
                "expected": " ".join(a_words[i1:i2]),
                "found": " ".join(b_words[j1:j2]),
            }
        )
    return diffs


def suggest_token_corrections(
    expected_tokens: List[str],
    found_tokens: List[str],
    *,
    min_score: int = 70,
) -> List[Dict[str, Optional[str]]]:
    """Suggest likely corrections for expected tokens using fuzzy matching.

    For each expected token, find the closest token in the found tokens.
    If the best score is >= min_score and not an exact match, return it.
    """
    suggestions: List[Dict[str, Optional[str]]] = []
    found_set = set(found_tokens)
    for token in expected_tokens:
        if token in found_set:
            continue
        if not found_tokens:
            suggestions.append({"expected": token, "suggested": None, "score": None})
            continue
        match = process.extractOne(token, found_tokens, scorer=fuzz.ratio)
        if match is None:
            suggestions.append({"expected": token, "suggested": None, "score": None})
            continue
        candidate, score, _ = match
        if int(score) >= min_score:
            suggestions.append({"expected": token, "suggested": candidate, "score": int(score)})
        else:
            suggestions.append({"expected": token, "suggested": None, "score": int(score)})
    return suggestions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect mismatches between model text and OCR text from an image."
    )
    parser.add_argument(
        "--text",
        "-t",
        default=MODEL_TEXT,
        help="Model's expected text description (default from script)",
    )
    parser.add_argument(
        "--image",
        "-i",
        default=IMAGE_PATH,
        help="Path to the screenshot/image to OCR",
    )
    parser.add_argument(
        "--threshold",
        "-th",
        type=int,
        default=SIMILARITY_THRESHOLD,
        help="Similarity threshold (0-100) to flag mismatch",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract PSM mode (default: 6)")
    parser.add_argument("--oem", type=int, default=3, help="Tesseract OEM mode (default: 3)")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing before OCR",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print normalized texts and detailed metrics",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return 2

    print("[INFO] Extracting text from image...")
    try:
        image_text_raw = extract_text_from_image(
            args.image,
            lang=args.lang,
            psm=args.psm,
            oem=args.oem,
            enable_preprocess=not args.no_preprocess,
        )
    except Exception as exc:
        print(f"[ERROR] OCR failed: {exc}")
        return 2

    if args.debug:
        print("[DEBUG] Raw OCR Text:")
        print(image_text_raw.strip())

    model_text_norm = normalize(args.text)
    image_text_norm = normalize(image_text_raw)

    metrics = compute_similarity_metrics(model_text_norm, image_text_norm)

    if args.debug:
        print("\n[DEBUG] Normalized Texts:")
        print("Model:", model_text_norm)
        print("Image:", image_text_norm)

    print("\n[RESULT] Similarity Metrics (0-100):")
    for name, value in metrics.items():
        print(f"- {name}: {value}")

    score = metrics["char_ratio"]
    if score < args.threshold:
        print("\n❌ Potential mismatch detected!")

        diffs = get_word_differences(model_text_norm, image_text_norm)
        if diffs:
            print("\n[DIFFERENCES]")
            for d in diffs:
                print(
                    f"- {d['type'].upper()} | Expected: '{d['expected']}' | Found: '{d['found']}'"
                )

        expected_tokens = tokenize(args.text)
        found_tokens = tokenize(image_text_raw)
        suggestions = suggest_token_corrections(expected_tokens, found_tokens)
        useful_suggestions = [s for s in suggestions if s.get("suggested")]
        if useful_suggestions:
            print("\n[SUGGESTIONS]")
            for s in useful_suggestions:
                print(
                    f"- '{s['expected']}' → '{s['suggested']}' (score: {s['score']})"
                )
        return 1
    else:
        print("\n✅ Text & Image look consistent.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
