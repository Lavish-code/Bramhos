# OCR-Validator  — OCR Validation & Text Matching Tool

A Python utility for comparing expected text against OCR-extracted text from images. Perfect for verifying OCR outputs, debugging text recognition, or automatically validating image and text file pairs.

---

## ✨ Features

- **Image Preprocessing**: Automatic grayscale conversion, contrast enhancement, and binarization for improved OCR accuracy
- **Tesseract Integration**: Leverages Tesseract OCR for robust text extraction
- **Smart Comparison**: Normalizes, tokenizes, and compares text with multiple similarity metrics
- **Detailed Analytics**: Computes character ratio, partial ratio, token set ratio, and more
- **Word-Level Diff**: Shows precise differences and fuzzy correction suggestions
- **Watch Mode**: Continuously monitors directories for new image/text pairs
- **Flexible Output**: Exit on mismatch or continue logging for batch processing

---

## 📋 Prerequisites

### Required Software

1. **Python 3.7+** (tested with Python 3.9–3.13)
2. **Tesseract OCR Engine**

### Installation Steps

**1. Install Python Dependencies**
```bash
pip install pillow pytesseract rapidfuzz
```

**2. Install Tesseract OCR**

**Windows:**
- Download from [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Add Tesseract to your system PATH, or configure `pytesseract.pytesseract.tesseract_cmd`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**3. Clone Repository (Optional)**
```bash
git clone https://github.com/Lavish-code/OCR-Validator .git
cd OCR-Validator 
```

---

## 🚀 Usage

### Single Image Validation

Validate a single image against expected text:

```bash
python ocr_validation.py --image path/to/image.png --text "Expected Text Here"
```

**Common Options:**

| Option | Shorthand | Default | Description |
|--------|-----------|---------|-------------|
| `--threshold` | `-th` | 80 | Minimum similarity score (0-100) |
| `--lang` | | eng | Tesseract language code |
| `--psm` | | 6 | Page segmentation mode |
| `--oem` | | 3 | OCR engine mode |
| `--no-preprocess` | | False | Skip image preprocessing |
| `--debug` | | False | Enable debug output |

### Watch Mode

Monitor a directory for new images and automatically validate them:

```bash
python ocr_validation.py --watch --watch-dir path/to/watch_folder
```

**Watch Mode Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--image-glob` | *.png,*.jpg,*.jpeg | Image file patterns (comma-separated) |
| `--text-exts` | .txt,.caption,.json | Sidecar file extensions |
| `--json-key` | text | JSON field containing text |
| `--interval` | 1.0 | Polling interval (seconds) |
| `--fail-on-mismatch` | False | Exit immediately on first mismatch |

---

## 📝 Examples

### Example 1: Basic Validation

```bash
python ocr_validation.py -i screenshots/output1.png -t "Hello, world!"
```

**Output:**
```
[RESULT] Similarity Metrics (0-100):
- char_ratio: 92
- partial_ratio: 94
- token_sort_ratio: 88
- token_set_ratio: 89

✅ Text & Image look consistent.
```

### Example 2: Mismatch Detection

```bash
python ocr_validation.py -i sample.png -t "Nike Air Shoes"
```

**Output:**
```
[RESULT] Similarity Metrics (0-100):
- char_ratio: 65

❌ Potential mismatch detected!

[DIFFERENCES]
- REPLACE | Expected: 'nike' | Found: 'nikee'
- DELETE  | Expected: 'air'  | Found: ''

[SUGGESTIONS]
- 'air' → 'ar' (score: 80)
```

### Example 3: Directory Monitoring

Monitor a folder with image/text pairs:

```bash
python ocr_validation.py --watch --watch-dir outputs --fail-on-mismatch
```

Expected structure:
```
outputs/
├── img1.png
├── img1.txt
├── img2.png
└── img2.json
```

---

## 📂 Project Structure

```
OCR-Validator /
│
├── ocr_validation.py    # Main validation script
├── README.md            # Documentation
└── tests/               # Test files (optional)
```

---

## 🔧 Configuration Tips

### Image Preprocessing
- **Clean images**: Use `--no-preprocess` if your images are already optimized
- **Poor quality**: Keep preprocessing enabled for scanned or low-quality images

### Similarity Threshold
- **Strict matching**: Use threshold ≥ 90 for critical applications
- **Fuzzy matching**: Use threshold 70-80 for more lenient validation
- **Very loose**: Use threshold < 70 for experimental setups

### JSON Sidecar Files
For nested JSON structures, use `--json-key` to specify the field:

```bash
python ocr_validation.py -i image.png --text-exts .json --json-key data.description
```

### Batch Processing
Use watch mode with logging for unattended batch validation:

```bash
python ocr_validation.py --watch --watch-dir batch_folder > validation.log 2>&1
```

---

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

- [ ] Add unit tests for normalization and fuzzy matching
- [ ] Support additional OCR engines (EasyOCR, PaddleOCR)
- [ ] Export results in CSV/JSON/HTML formats
- [ ] Multi-language support and custom dictionaries
- [ ] GUI interface for easier operation
- [ ] Parallel processing for batch operations

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with tests

---

## 📄 License

This project is open source. Check the repository for license details.

---

## 🐛 Troubleshooting

**Issue**: `TesseractNotFoundError`
- **Solution**: Ensure Tesseract is installed and in your PATH

**Issue**: Low accuracy scores
- **Solution**: Try adjusting `--psm` values (3 for fully automatic, 6 for uniform block)

**Issue**: Slow processing
- **Solution**: Use `--no-preprocess` or increase `--interval` in watch mode

---

## 📞 Support

For issues, questions, or feature requests, please visit the [GitHub repository](https://github.com/Lavish-code/OCR-Validator ).

---

**Made with ❤️ for the OCR communit
