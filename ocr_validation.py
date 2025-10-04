import pytesseract
from PIL import Image
import re
from rapidfuzz import fuzz
from difflib import SequenceMatcher

# ---------- CONFIGURATION ----------
MODEL_TEXT = "Searching for NikeAir shoes on Myntra"
IMAGE_PATH = "result.png"   # path to your image result
SIMILARITY_THRESHOLD = 80   # below this, flag mismatch
# ----------------------------------


def extract_text_from_image(image_path: str) -> str:
    """Extract raw text from image using OCR."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


def normalize(text: str) -> str:
    """Lowercase and remove punctuation/spaces."""
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower()).strip()


def get_word_differences(a: str, b: str):
    """Find word-level differences between two texts."""
    a_words, b_words = a.split(), b.split()
    sm = SequenceMatcher(None, a_words, b_words)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            diffs.append({
                "type": tag,
                "expected": " ".join(a_words[i1:i2]),
                "found": " ".join(b_words[j1:j2])
            })
    return diffs


def main():
    # Step 1: OCR
    print("[INFO] Extracting text from image...")
    image_text_raw = extract_text_from_image(IMAGE_PATH)
    print("Extracted Image Text:", image_text_raw.strip())

    # Step 2: Normalize
    model_text_norm = normalize(MODEL_TEXT)
    image_text_norm = normalize(image_text_raw)

    print("\n[DEBUG] Normalized Texts:")
    print("Model  :", model_text_norm)
    print("Image  :", image_text_norm)

    # Step 3: Similarity Check
    score = fuzz.ratio(model_text_norm, image_text_norm)
    print("\n[RESULT] Similarity Score:", score)

    if score < SIMILARITY_THRESHOLD:
        print("❌ Potential mismatch detected!")

        # Step 4: Show word differences
        diffs = get_word_differences(model_text_norm, image_text_norm)
        print("\n[DIFFERENCES]")
        for d in diffs:
            print(f"- {d['type'].upper()} | Expected: '{d['expected']}' | Found: '{d['found']}'")
    else:
        print("✅ Text & Image look consistent.")


if __name__ == "__main__":
    main()
