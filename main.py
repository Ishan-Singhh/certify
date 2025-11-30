# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio

# image / OCR / face libs
from deepface import DeepFace
import cv2
from PIL import Image, ImageOps
import pytesseract
import re
from rapidfuzz import fuzz

# ---------- Utility functions (adapted from your notebook) ----------

def save_upload_to_temp(upload_file: UploadFile) -> str:
    """Save an UploadFile to a temporary file and return its path."""
    suffix = os.path.splitext(upload_file.filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload_file.file, tmp)
        tmp.flush()
        tmp.close()
    finally:
        upload_file.file.close()
    return tmp.name

def is_blurry_path(image_path: str, threshold: float = 1000.0) -> Tuple[bool, float]:
    """Return (is_blurry, variance). True if variance < threshold."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image for blur check")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    return variance < threshold, variance

def is_blank_path(image_path: str, threshold: float = 10.0) -> Tuple[bool, float]:
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
    # psm 3 is what you used; tune as needed
    raw = pytesseract.image_to_string(resized_image, config="--psm 3", lang=lang)
    return normalize_text(raw)

def verify_photo_paths(cert_path: str, selfie_path: str, model_name: str = "VGG-Face", detector_backend: str = "opencv"):
    """Return DeepFace.verify dict (may take time)."""
    # DeepFace.verify returns a dict with 'verified' boolean and distance etc.
    return DeepFace.verify(img1_path=cert_path, img2_path=selfie_path, model_name=model_name, detector_backend=detector_backend)

def verify_name_in_text(ocr_text: str, name: str, threshold: int = 80) -> dict:
    """Return (match_bool, avg_score, candidates) for name verification."""
    t = ocr_text.replace('\n', ' ')
    t = re.sub(r'\s{2,}', ' ', t)
    name_candidates = []

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

    # fallback: try to pick capitalized words sequences
    if not name_candidates:
        # crude fallback: find sequences of 2-3 capitalized words
        caps = re.findall(r'\b[A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})?\b', ocr_text)
        name_candidates.extend(caps[:3])

    # compute average partial ratio
    scores = []
    for cand in name_candidates:
        if cand:
            scores.append(fuzz.partial_ratio(name.lower(), cand.lower()))
    avg = sum(scores) / len(scores) if scores else 0.0
    matched = avg > threshold
    return {"matched": matched, "avg_score": avg, "candidates": name_candidates, "scores": scores}

def verify_address_in_text(ocr_text: str, address: str, threshold: int = 80) -> dict:
    score = fuzz.partial_ratio(address.lower(), ocr_text.lower())
    return {"matched": score > threshold, "score": score}

# ---------- FastAPI app ----------

app = FastAPI(title="Certificate Verification API")

# Allow CORS for local testing (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# Health
@app.get("/health")
async def health():
    return {"status": "ok"}

# Single endpoint: upload files and run verification
@app.post("/verify")
async def verify(
    cert: UploadFile = File(...),
    selfie: UploadFile = File(...),
    name: str = "",
    address: str = "",
    blur_threshold: float = 1000.0,
    blank_threshold: float = 10.0,
    name_threshold: int = 80,
    address_threshold: int = 80,
):
    # save uploads
    cert_path = save_upload_to_temp(cert)
    selfie_path = save_upload_to_temp(selfie)

    try:
        # run checks concurrently in threadpool to avoid blocking event loop
        loop = asyncio.get_running_loop()

        # blur and blank checks
        blur_task = loop.run_in_executor(executor, is_blurry_path, cert_path, blur_threshold)
        blank_task = loop.run_in_executor(executor, is_blank_path, cert_path, blank_threshold)
        ocr_task = loop.run_in_executor(executor, extract_text_path, cert_path)
        face_task = loop.run_in_executor(executor, verify_photo_paths, cert_path, selfie_path)

        blur_res, blank_res, ocr_text, face_res = await asyncio.gather(blur_task, blank_task, ocr_task, face_task)

        # name/address verification (run locally)
        name_ver = verify_name_in_text(ocr_text, name or "", threshold=name_threshold) if name else {"matched": None}
        address_ver = verify_address_in_text(ocr_text, address or "", threshold=address_threshold) if address else {"matched": None}

        response = {
            "checks": {
                "is_blurry": bool(blur_res[0]),
                "blur_variance": blur_res[1],
                "is_blank": bool(blank_res[0]),
                "blank_stddev": blank_res[1],
            },
            "face_verification": face_res,  # full DeepFace response
            "ocr_text": ocr_text,
            "name_verification": name_ver,
            "address_verification": address_ver,
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup temp files
        try:
            os.remove(cert_path)
        except Exception:
            pass
        try:
            os.remove(selfie_path)
        except Exception:
            pass

# Smaller convenience endpoint: just OCR
@app.post("/ocr")
async def ocr_only(cert: UploadFile = File(...)):
    cert_path = save_upload_to_temp(cert)
    try:
        loop = asyncio.get_running_loop()
        ocr_text = await loop.run_in_executor(executor, extract_text_path, cert_path)
        return {"ocr_text": ocr_text}
    finally:
        try:
            os.remove(cert_path)
        except Exception:
            pass
