# app.py
import os
import re
import tempfile
import shutil
import asyncio
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from PIL import Image, ImageOps
import cv2
import pytesseract
from deepface import DeepFace
from rapidfuzz import fuzz

# ---------- Config ----------
# Number of threads for blocking CPU work (OCR / DeepFace / OpenCV)
EXECUTOR_WORKERS = 4

# Default thresholds (internal, fixed)
BLUR_THRESHOLD = 1000.0
BLANK_STDDEV_THRESHOLD = 10.0
NAME_MATCH_THRESHOLD = 80
ADDRESS_MATCH_THRESHOLD = 80

# ---------- App ----------
app = FastAPI(title="Certificate Verification ")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)

# ---------- Helper utilities ----------

def download_url_to_temp(url: str, suffix: str = ".jpg", timeout: int = 15) -> str:
    """Download an image from a URL to a temporary file and return its path."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Failed to download {url}: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(resp.raw, tmp)
        tmp.flush()
        tmp.close()
    finally:
        resp.close()
    return tmp.name

def cleanup_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass

def is_blurry_path(image_path: str, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """Return (is_blurry, variance). True if variance < threshold."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image for blur check")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    return variance < threshold, variance

def is_blank_path(image_path: str, threshold: float = BLANK_STDDEV_THRESHOLD) -> Tuple[bool, float]:
    """Return (is_blank, stddev). True if stddev < threshold."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image for blank check")
    mean, stddev = cv2.meanStdDev(img)
    std = float(stddev[0][0])
    return std < threshold, std

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s/.-]', '', s)
    return s.lower()

def extract_text_path(image_path: str, lang: str = "eng+hin") -> str:
    """Open, preprocess and run pytesseract and return normalized text."""
    image = Image.open(image_path)
    gray_image = ImageOps.grayscale(image)
    scale_factor = 2
    resized_image = gray_image.resize(
        (gray_image.width * scale_factor, gray_image.height * scale_factor),
        resample=Image.LANCZOS
    )
    raw = pytesseract.image_to_string(resized_image, config="--psm 3", lang=lang)
    return normalize_text(raw)

def verify_photo_paths(cert_path: str, selfie_path: str, model_name: str = "VGG-Face", detector_backend: str = "retinaface") -> Dict[str, Any]:
    """Run DeepFace.verify and return the result dict."""
    # DeepFace.verify returns a dict with 'verified' boolean and additional metrics
    return DeepFace.verify(img1_path=cert_path, img2_path=selfie_path, model_name=model_name, detector_backend=detector_backend)

def verify_name_in_text(ocr_text: str, name: str, threshold: int = NAME_MATCH_THRESHOLD) -> Dict[str, Any]:
    """Return structured info about name matching."""
    t = ocr_text.replace('\n', ' ')
    t = re.sub(r'\s{2,}', ' ', t)
    name_candidates = []

    # Try several heuristics to extract name candidates
    m = re.search(r'This\s+is\s+to\s+certify\s+that\s+(.{2,80}?)\s+(?:son|daughter)\b', t, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r'\bof\b.*$', '', candidate, flags=re.IGNORECASE).strip()
        name_candidates.append(candidate)

    m2 = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+(?:son|daughter)\s+of', t, flags=re.IGNORECASE)
    if m2:
        name_candidates.append(m2.group(1).strip())

    m3 = re.search(r'([A-Z][\w\s]{2,60}?)\s+(?:S/O|D/O|son|daughter)\b', t, flags=re.IGNORECASE)
    if m3:
        name_candidates.append(m3.group(1).strip())

    # fallback: pick sequences of 2-3 capitalized words
    if not name_candidates:
        caps = re.findall(r'\b[A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})?\b', ocr_text)
        name_candidates.extend(caps[:3])

    scores = []
    for cand in name_candidates:
        if cand:
            scores.append(fuzz.partial_ratio(name.lower(), cand.lower()))
    avg = float(sum(scores) / len(scores)) if scores else 0.0
    matched = avg > threshold
    return {"matched": matched, "avg_score": avg, "candidates": name_candidates, "scores": scores}

def verify_address_in_text(ocr_text: str, address: str, threshold: int = ADDRESS_MATCH_THRESHOLD) -> Dict[str, Any]:
    score = fuzz.partial_ratio(address.lower(), ocr_text.lower())
    return {"matched": score > threshold, "score": score}

# ---------- API Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/verify")
async def verify(
    cert_url: str,
    selfie_url: str,
    name: str,
    address: str,
):
    """
    Verify certificate and selfie provided as URLs.
    Request parameters:
      - cert_url (str): URL to the certificate image (Cloudinary or any HTTP(S) URL)
      - selfie_url (str): URL to the selfie image
      - name (str, optional): Name to verify against OCR (e.g., "Ishan Singh")
      - address (str, optional): Address to verify against OCR
    """
    # Download images
    try:
        cert_tmp = download_url_to_temp(cert_url)
        selfie_tmp = download_url_to_temp(selfie_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        loop = asyncio.get_running_loop()

        # Run blocking tasks in threadpool concurrently
        blur_task = loop.run_in_executor(executor, is_blurry_path, cert_tmp)
        blank_task = loop.run_in_executor(executor, is_blank_path, cert_tmp)
        ocr_task = loop.run_in_executor(executor, extract_text_path, cert_tmp)
        face_task = loop.run_in_executor(executor, verify_photo_paths, cert_tmp, selfie_tmp)

        blur_res, blank_res, ocr_text, face_res = await asyncio.gather(blur_task, blank_task, ocr_task, face_task)

        # Name and address verification (if provided)
        name_ver = verify_name_in_text(ocr_text, name) if name else {"matched": None}
        address_ver = verify_address_in_text(ocr_text, address) if address else {"matched": None}

        response = {
            "checks": {
                "is_blurry": bool(blur_res[0]),
                "blur_variance": blur_res[1],
                "is_blank": bool(blank_res[0]),
                "blank_stddev": blank_res[1],
            },
            "face_verification": face_res,  # raw DeepFace response (contains 'verified' boolean and distances)
            "ocr_text": ocr_text,
            "name_verification": name_ver,
            "address_verification": address_ver,
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        cleanup_file(cert_tmp)
        cleanup_file(selfie_tmp)

@app.post("/ocr")
async def ocr_only(cert_url: str):
    """
    OCR-only endpoint that accepts an image URL and returns normalized OCR text.
    """
    try:
        cert_tmp = download_url_to_temp(cert_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        loop = asyncio.get_running_loop()
        ocr_text = await loop.run_in_executor(executor, extract_text_path, cert_tmp)
        return {"ocr_text": ocr_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    finally:
        cleanup_file(cert_tmp)
