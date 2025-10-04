import argparse
import os
import sys
import re
import time
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pytesseract
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher

# ---------- CONFIGURATION ----------
SIMILARITY_THRESHOLD = 80   # 0-100; below this, flag mismatch
# ----------------------------------

def preprocess_image(image_path: str, enable_preprocess: bool = True) -> Image.Image:
    """Open and optionally preprocess the image to improve OCR accuracy."""
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unable to open image (unrecognized format): {image_path}") from exc

    if not enable_preprocess:
        return image

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.5)

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
    """Extract raw text from image using Tesseract OCR."""
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
    """Normalize text for fair matching."""
    if text is None:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"[\-_./]+", " ", text)
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
    """Find word-level differences between two normalized texts."""
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
    """Suggest likely corrections for expected tokens using fuzzy matching."""
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


def read_sidecar_text_for_image(
    image_path: str,
    *,
    text_exts: Tuple[str, ...] = (".txt", ".caption", ".json"),
    json_key: str = "text",
) -> Optional[str]:
    """Read sidecar text for an image if available.

    Attempts the following for the same basename as the image:
    - <basename>.txt or .caption: read raw text
    - <basename>.json: read JSON and extract `json_key`
    Returns None if nothing usable is found.
    """
    image_path_obj = Path(image_path)
    base = image_path_obj.with_suffix("")
    for ext in text_exts:
        candidate = base.with_suffix(ext)
        if not candidate.exists():
            continue
        try:
            if candidate.suffix.lower() == ".json":
                with candidate.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                value = data
                for part in json_key.split(".") if json_key else []:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                if isinstance(value, str) and value.strip():
                    return value
            else:
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
        except Exception:
            # Ignore sidecar read errors; treat as missing
            continue
    return None


def validate_text_vs_image(
    expected_text: str,
    image_path: str,
    *,
    threshold: int,
    lang: str,
    psm: int,
    oem: int,
    enable_preprocess: bool,
    debug: bool,
    quiet: bool = False,
) -> Tuple[bool, Dict[str, int]]:
    """Run OCR on the image and compare with expected text.

    Returns (is_match, metrics)
    """
    if not os.path.exists(image_path):
        if not quiet:
            print(f"[ERROR] Image not found: {image_path}")
        return False, {}

    try:
        image_text_raw = extract_text_from_image(
            image_path,
            lang=lang,
            psm=psm,
            oem=oem,
            enable_preprocess=enable_preprocess,
        )
    except Exception as exc:
        if not quiet:
            print(f"[ERROR] OCR failed for {image_path}: {exc}")
        return False, {}

    if debug and not quiet:
        print("[DEBUG] Raw OCR Text:")
        print(image_text_raw.strip())

    model_text_norm = normalize(expected_text)
    image_text_norm = normalize(image_text_raw)

    metrics = compute_similarity_metrics(model_text_norm, image_text_norm)

    if not quiet:
        print("\n[RESULT] Similarity Metrics (0-100):")
        for name, value in metrics.items():
            print(f"- {name}: {value}")

    score = metrics.get("char_ratio", 0)
    is_match = score >= threshold

    if not is_match and not quiet:
        print("\n❌ Potential mismatch detected!")

        diffs = get_word_differences(model_text_norm, image_text_norm)
        if diffs:
            print("\n[DIFFERENCES]")
            for d in diffs:
                print(
                    f"- {d['type'].upper()} | Expected: '{d['expected']}' | Found: '{d['found']}'"
                )

        expected_tokens = tokenize(expected_text)
        found_tokens = tokenize(image_text_raw)
        suggestions = suggest_token_corrections(expected_tokens, found_tokens)
        useful_suggestions = [s for s in suggestions if s.get("suggested")]
        if useful_suggestions:
            print("\n[SUGGESTIONS]")
            for s in useful_suggestions:
                print(f"- '{s['expected']}' → '{s['suggested']}' (score: {s['score']})")

    if is_match and not quiet:
        print("\n✅ Text & Image look consistent.")

    return is_match, metrics


def watch_and_validate(
    *,
    watch_dir: str,
    image_globs: List[str],
    text_exts: Tuple[str, ...],
    json_key: str,
    interval_s: float,
    threshold: int,
    lang: str,
    psm: int,
    oem: int,
    enable_preprocess: bool,
    debug: bool,
    fail_on_mismatch: bool,
) -> int:
    """Continuously watch a directory for new/updated images and validate with sidecar text."""
    directory = Path(watch_dir)
    if not directory.exists() or not directory.is_dir():
        print(f"[ERROR] Watch directory not found or not a dir: {watch_dir}")
        return 2

    print(
        f"[INFO] Watching '{directory.resolve()}' for images: {', '.join(image_globs)} with sidecars {', '.join(text_exts)}"
    )
    if json_key:
        print(f"[INFO] JSON key for text: '{json_key}'")
    print("[INFO] Press Ctrl+C to stop.")

    processed: Dict[str, float] = {}
    any_mismatch = False

    try:
        while True:
            matched_images: Set[str] = set()
            for pattern in image_globs:
                for path in glob.glob(str(directory / pattern)):
                    matched_images.add(os.path.abspath(path))

            for image_path in sorted(matched_images):
                try:
                    mtime = os.path.getmtime(image_path)
                except FileNotFoundError:
                    continue

                last = processed.get(image_path)
                if last is not None and mtime <= last:
                    continue  # unchanged

                # Try to read sidecar text
                expected_text = read_sidecar_text_for_image(
                    image_path, text_exts=text_exts, json_key=json_key
                )
                if expected_text is None:
                    if debug:
                        print(f"[DEBUG] No sidecar text found for {image_path}")
                    # Do not mark as processed so we can retry next loop
                    continue

                print(f"\n[FILE] Validating {image_path}")
                is_match, metrics = validate_text_vs_image(
                    expected_text,
                    image_path,
                    threshold=threshold,
                    lang=lang,
                    psm=psm,
                    oem=oem,
                    enable_preprocess=enable_preprocess,
                    debug=debug,
                    quiet=False,
                )

                processed[image_path] = mtime
                if not is_match:
                    any_mismatch = True
                    if fail_on_mismatch:
                        print("[INFO] Exiting due to mismatch and --fail-on-mismatch set.")
                        return 1

            time.sleep(max(0.1, float(interval_s)))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped watching.")

    return 1 if any_mismatch else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect mismatches between model text and OCR text from an image."
    )
    # Either provide --text and --image for a one-off check, or use --watch mode
    parser.add_argument("--text", "-t", default=None, help="Model's expected text")
    parser.add_argument("--image", "-i", default=None, help="Path to image")
    parser.add_argument("--threshold", "-th", type=int, default=SIMILARITY_THRESHOLD, help="Threshold")
    parser.add_argument("--lang", default="eng", help="Tesseract language")
    parser.add_argument("--psm", type=int, default=6, help="PSM mode")
    parser.add_argument("--oem", type=int, default=3, help="OEM mode")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    # Watch mode options
    parser.add_argument("--watch", action="store_true", help="Watch a directory for new image+text pairs")
    parser.add_argument("--watch-dir", default=".", help="Directory to watch for outputs")
    parser.add_argument(
        "--image-glob",
        default="*.png,*.jpg,*.jpeg",
        help="Comma-separated image glob patterns inside watch dir",
    )
    parser.add_argument(
        "--text-exts",
        default=".txt,.caption,.json",
        help="Comma-separated sidecar text extensions to try",
    )
    parser.add_argument(
        "--json-key",
        default="text",
        help="JSON key path (dot-separated) to extract text from sidecar JSON",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval seconds in watch mode")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit immediately with non-zero status when a mismatch is detected in watch mode",
    )

    args = parser.parse_args(argv)

    if args.watch:
        image_globs = [p.strip() for p in args.image_glob.split(",") if p.strip()]
        text_exts = tuple(e if e.startswith(".") else f".{e}" for e in [t.strip() for t in args.text_exts.split(",") if t.strip()])
        return watch_and_validate(
            watch_dir=args.watch_dir,
            image_globs=image_globs,
            text_exts=text_exts,
            json_key=args.json_key,
            interval_s=args.interval,
            threshold=args.threshold,
            lang=args.lang,
            psm=args.psm,
            oem=args.oem,
            enable_preprocess=not args.no_preprocess,
            debug=args.debug,
            fail_on_mismatch=args.fail_on_mismatch,
        )

    # Non-watch single run requires both text and image
    if not args.text or not args.image:
        print("[ERROR] Either use --watch or provide both --text and --image.")
        parser.print_help()
        return 2

    print("[INFO] Extracting text from image...")
    is_match, _ = validate_text_vs_image(
        args.text,
        args.image,
        threshold=args.threshold,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        enable_preprocess=not args.no_preprocess,
        debug=args.debug,
        quiet=False,
    )
    return 0 if is_match else 1


if __name__ == "__main__":
    sys.exit(main())
