"""
Easter Item Checker v1.0
========================
Simplified catalog image sourcing for seasonal/Easter items.
For each item, searches JETS shelf scans and community photos for a clear
photo, crops the item, removes the background via gpt-image-1, and uploads
a catalog-ready image.

Differences from Electronics Packaging Checker:
  - No packaging check (skip entirely -- process ALL items)
  - No eBay reference lookup
  - Matches items to photos via BOTH MSID and DD_SIC
  - Community photos from two files: MSID-based and DD_SIC-based
  - Simplified scoring (clear product photo, not packaging-specific)

Flow per item:
  1. Collect candidate photos: JETS by dd_sic, community by dd_sic + msid
  2. Score each candidate for clarity, lighting, product visibility
  3. Two-pass crop to isolate the product
  4. gpt-image-1 edit: remove background -> pure white
  5. Upload to imgBB for public URL
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import zipfile
import logging
from pathlib import Path
from collections import defaultdict

import requests
from PIL import Image, ImageEnhance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    logger.info("pillow-heif registered")
except ImportError:
    logger.warning("pillow-heif not found -- HEIF images will be skipped")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MIN_SCORE_THRESHOLD = 6
EARLY_STOP_SCORE = 8
MIN_BRIGHTNESS = 60
MAX_VERIFIED_COMMUNITY = 5
MAX_PHOTO_URLS_TO_CHECK = 3


# --------------------------------------------------------------------------
# OpenAI helpers
# --------------------------------------------------------------------------

def _openai_vision(prompt, jpeg_bytes_list, openai_key, max_tokens=500, detail="high"):
    content = [{"type": "text", "text": prompt}]
    for jpeg_bytes in jpeg_bytes_list:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
        img.thumbnail((1500, 1500), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}
        })
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {openai_key}"},
        json={"model": "gpt-4o",
              "messages": [{"role": "user", "content": content}],
              "max_tokens": max_tokens, "temperature": 0.1},
        timeout=45
    )
    if resp.status_code != 200:
        logger.warning(f"  GPT-4o returned {resp.status_code}: {resp.text[:200]}")
        return None
    result = resp.json()['choices'][0]['message']['content']
    return result.strip() if result else None


def _parse_json(raw):
    if not raw:
        return None
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except Exception:
        return None


def _img_to_jpeg(img, quality=85, max_side=1500):
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def get_brightness(img):
    try:
        import numpy as np
        return float(np.array(img.convert("L")).mean())
    except Exception:
        return 128.0


# --------------------------------------------------------------------------
# Image download utilities
# --------------------------------------------------------------------------

def download_image_as_jpeg(url, timeout=15, max_dimension=1024):
    try:
        resp = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get('Content-Type', '').lower()
        if 'html' in content_type or 'text' in content_type:
            return None
        is_heif = (
            'heif' in content_type or 'heic' in content_type or
            url.lower().endswith(('.heif', '.heic')) or
            'full_image.heic' in url.lower()
        )
        if is_heif:
            try:
                import pillow_heif
                heif_file = pillow_heif.read_heif(content)
                img = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data,
                    "raw", heif_file.mode, heif_file.stride
                )
            except (ImportError, Exception):
                return None
        else:
            try:
                img = Image.open(io.BytesIO(content))
            except Exception:
                return None
        if img.width < 100 or img.height < 100:
            return None
        img = img.convert('RGB')
        if img.width > max_dimension or img.height > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()
    except Exception:
        return None


def download_image_as_pil(url, timeout=30):
    try:
        resp = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        return img.convert('RGB')
    except Exception:
        return None


# --------------------------------------------------------------------------
# imgBB upload
# --------------------------------------------------------------------------

def upload_to_imgbb(filepath, imgbb_key, item_name=''):
    if not imgbb_key:
        return (None, None)
    try:
        with open(filepath, 'rb') as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')
        safe_name = "".join(c if c.isalnum() or c in '-_ ' else '_' for c in item_name[:60])
        resp = requests.post("https://api.imgbb.com/1/upload",
                             data={"key": imgbb_key, "image": b64_data,
                                   "name": safe_name or os.path.basename(filepath)},
                             timeout=30)
        if resp.status_code == 200:
            data = resp.json().get('data', {})
            display_url = data.get('display_url', data.get('url', ''))
            delete_url = data.get('delete_url', '')
            logger.info(f"    [UPLOAD] Public URL: {display_url[:80]}...")
            return (display_url, delete_url)
        else:
            logger.warning(f"    [UPLOAD] imgBB returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"    [UPLOAD] imgBB upload failed: {e}")
    return (None, None)


# --------------------------------------------------------------------------
# Google reference image search via SerpAPI
# --------------------------------------------------------------------------

def search_reference_images(item_name, serpapi_key, max_results=5):
    """Search Google Images for product reference photos (ground truth for text check)."""
    if not serpapi_key:
        return []
    clean_name = re.sub(r'\s*\([^)]*\)', '', item_name).strip()
    query = f"{clean_name} product packaging"
    logger.info(f"    [REF_SEARCH] Searching: {query}")
    try:
        resp = requests.get('https://serpapi.com/search', params={
            'engine': 'google_images', 'q': query,
            'api_key': serpapi_key, 'num': max_results * 2,
        }, timeout=30)
        resp.raise_for_status()
        results = resp.json().get('images_results', [])
    except Exception as e:
        logger.info(f"    [REF_SEARCH] Search failed: {e}")
        return []
    if not results:
        logger.info(f"    [REF_SEARCH] No results found")
        return []
    ref_images = []
    for img_result in results:
        if len(ref_images) >= max_results:
            break
        url = img_result.get('original') or img_result.get('thumbnail')
        if not url:
            continue
        jpeg_bytes = download_image_as_jpeg(url, timeout=10, max_dimension=768)
        if jpeg_bytes:
            ref_images.append(jpeg_bytes)
    logger.info(f"    [REF_SEARCH] Downloaded {len(ref_images)} reference images")
    return ref_images


# --------------------------------------------------------------------------
# Final text check + blur against reference images
# --------------------------------------------------------------------------

FINAL_TEXT_CHECK_PROMPT = """I am providing you TWO images:

REFERENCE IMAGE (first image): Shows the REAL product from a Google image search.
OUTPUT IMAGE (second image): An AI-edited catalog photo we just created.

Compare EVERY piece of text on the OUTPUT image against the REFERENCE image.

RULES:
- Every word on the output MUST match the reference exactly
- Missing text is OK (less text than reference is acceptable)
- But text that IS present must be spelled correctly and match the reference
- Misspelled text = PROBLEM (e.g. "EDOS" instead of "EGGS", "ressalable" instead of "resealable")
- Wrong words = PROBLEM (e.g. "ESS SOUND" instead of "VIA APP")
- Hallucinated text (words not on reference at all) = PROBLEM
- Blurry text that roughly matches = OK, not a problem
- Very small text that is hard to read = OK if it roughly matches

For each problem, describe WHERE on the packaging and what it should say.

If all visible text matches (or is too blurry/small to read), respond: NO_ISSUES

Otherwise respond with a list:
- Location: "WRONG_TEXT" should be "CORRECT_TEXT" or should be blurred
"""


def final_text_check_and_blur(generated_img, ref_images, openai_key):
    """Compare output text against Google reference images.
    If any text is wrong/misspelled, blur those specific areas."""
    if not ref_images or not openai_key:
        return generated_img
    gen_jpeg = _img_to_jpeg(generated_img)
    try:
        all_images = [ref_images[0], gen_jpeg]
        raw = _openai_vision(FINAL_TEXT_CHECK_PROMPT, all_images, openai_key,
                             max_tokens=300, detail="high")
        if not raw or 'NO_ISSUES' in raw:
            logger.info(f"    [FINAL_TEXT] All text matches reference")
            return generated_img
        logger.info(f"    [FINAL_TEXT] Text issues found: {raw[:150]}...")
    except Exception as e:
        logger.debug(f"    [FINAL_TEXT] Check failed: {e}")
        return generated_img
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        img = generated_img.convert('RGBA')
        max_side = max(img.width, img.height)
        square = Image.new("RGBA", (max_side, max_side), (255, 255, 255, 255))
        square.paste(img, ((max_side - img.width) // 2, (max_side - img.height) // 2))
        square.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        square.save(buf, format="PNG")
        buf.seek(0)
        buf.name = "source.png"
        blur_prompt = (
            f"This is a product catalog image. Some text on the product is "
            f"incorrect or misspelled. BLUR (make unreadable) ONLY the following "
            f"text areas while keeping everything else exactly the same:\n\n{raw}\n\n"
            f"Do NOT change anything else. Do NOT remove, replace, or rewrite "
            f"the text -- just make those specific areas blurry/unreadable. "
            f"Keep all large correct text, product images, logos, and colors exactly as they are."
        )
        response = client.images.edit(
            model="gpt-image-1", image=buf, prompt=blur_prompt, size="1024x1024"
        )
        if response.data and response.data[0].b64_json:
            img_data = base64.b64decode(response.data[0].b64_json)
            fixed = Image.open(io.BytesIO(img_data)).convert("RGB")
            logger.info(f"    [FINAL_TEXT] Bad text blurred successfully")
            return fixed
        elif response.data and response.data[0].url:
            img_resp = requests.get(response.data[0].url, timeout=30)
            if img_resp.status_code == 200:
                fixed = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                logger.info(f"    [FINAL_TEXT] Bad text blurred successfully")
                return fixed
    except Exception as e:
        logger.info(f"    [FINAL_TEXT] Blur fix failed: {e}")
    return generated_img


# --------------------------------------------------------------------------
# JETS zip extraction
# --------------------------------------------------------------------------

def get_product_filename(shelf_tag_id, tag_product_associations_json):
    try:
        associations = json.loads(tag_product_associations_json)
        for a in associations:
            if a['first'] == shelf_tag_id:
                return f"product-{a['second']}.heif"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def download_and_extract_from_zip(product_zip_url, product_filename, timeout=30):
    try:
        resp = requests.get(product_zip_url, timeout=timeout)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if product_filename in names:
                return zf.read(product_filename)
            heic_name = product_filename.replace('.heif', '.heic')
            if heic_name in names:
                return zf.read(heic_name)
    except Exception as e:
        logger.debug(f"  Zip extraction failed: {e}")
    return None


def convert_heif_to_jpeg(image_bytes):
    try:
        import pillow_heif
        heif_file = pillow_heif.read_heif(image_bytes)
        img = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data,
            "raw", heif_file.mode, heif_file.stride
        )
    except ImportError:
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return None
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=85)
    return buf.getvalue()


# --------------------------------------------------------------------------
# GPT-4o prompts (simplified for general items, not electronics-specific)
# --------------------------------------------------------------------------

SCORE_PROMPT = """You are scoring a retail store photo for catalog usability.

Item we are looking for: "{item_name}"

Score this photo from 1 to 10:
- Is the item clearly visible and identifiable? (most important)
- Is the photo well-lit?
- Is the shot at a reasonable angle?
- Is the item large enough in the frame to crop cleanly?

9-10: Perfect -- item clear, well-lit, straight-on, easy to crop
7-8:  Good -- item visible with minor issues (slight angle, small)
5-6:  Marginal -- item present but dark/angled/partial/far away
1-4:  Poor -- can't identify the item, too dark, wrong item, unusable

Respond ONLY with valid JSON, no markdown:
{{"score": 1-10, "item_visible": true or false, "reason": "one sentence", "usable": true or false}}
"""

ITEM_LOCATION_CHECK_PROMPT = """This is an item location photo taken for a retail catalog.

Does this clearly show "{item_name}", well-lit and suitable for a product catalog?

Respond ONLY with valid JSON, no markdown:
{{"suitable": true or false, "confidence": "high" or "medium" or "low", "reason": "one sentence"}}
"""

LOCATE_COARSE_PROMPT = """Find the item "{item_name}" in this retail shelf/store photo.
Return a loose bounding box around the general region where the item is.

Respond ONLY with valid JSON:
If found: {{"found": true, "x1_pct": 0-100, "y1_pct": 0-100, "x2_pct": 0-100, "y2_pct": 0-100, "confidence": "high" or "medium" or "low"}}
If not found: {{"found": false}}
"""

LOCATE_PRECISE_PROMPT = """This is a zoomed-in photo of a retail shelf/display. Find the EXACT edges of
"{item_name}". Be tight -- exclude neighboring items, shelf labels, and price tags.

Respond ONLY with valid JSON, no markdown:
If found: {{"found": true, "x1_pct": 0-100, "y1_pct": 0-100, "x2_pct": 0-100, "y2_pct": 0-100, "confidence": "high" or "medium" or "low"}}
If not found: {{"found": false}}
"""

ORIENTATION_CHECK_PROMPT = """Look at this product photo.

Is the product oriented correctly (right-side up), or is it upside down,
rotated 90 clockwise, or rotated 90 counter-clockwise?

CLUES to check orientation:
- Text on the product/packaging: is it readable or upside down / sideways?
- Brand logos: are they right-side up?
- Product images: are they oriented normally?

Respond ONLY with valid JSON, no markdown:
{{"orientation": "correct", "confidence": "high"}}
or
{{"orientation": "upside_down", "confidence": "high"}}
or
{{"orientation": "rotated_90_cw", "confidence": "high"}}
or
{{"orientation": "rotated_90_ccw", "confidence": "high"}}
"""

CROP_QUALITY_PROMPT = """You are evaluating whether this cropped product photo is CLEAR ENOUGH
to use as a catalog image after background removal.

PASS (respond usable: true) if:
- The product is mostly visible and recognizable
- The product can be clearly identified from this image
- Minor angle, slight blur, or partial obstruction is OK

FAIL (respond usable: false) if:
- The product is severely blurry or out of focus
- The product is mostly obscured by other items, hands, or shadows
- You cannot tell what product this is from the image
- The crop missed the product entirely or got the wrong item
- The image is too dark to make out the product

Respond ONLY with valid JSON:
{{"usable": true, "reason": "one sentence"}}
or
{{"usable": false, "reason": "one sentence"}}
"""

GPT_IMAGE_EDIT_PROMPT = """You are a photo editor. Your ONLY job is to remove the background from this product photo and replace it with PURE WHITE.

Product: {item_name}

WHAT TO DO:
- Replace the entire background (shelf, store, hands, surfaces) with PURE WHITE (#FFFFFF)
- If multiple copies of the product are visible, keep ONLY the single most front-facing unit and remove the rest
- REMOVE any visible price tags, shelf labels, sale signs, or barcodes that are NOT part of the product
- Center the single product on the white background

CRITICAL -- PRESERVE EXACTLY:
- Every element of the product must stay in the EXACT same position
- The perspective, tilt, and rotation must remain UNCHANGED
- The proportions and aspect ratio must remain UNCHANGED
- All colors must remain UNCHANGED
- Do NOT rearrange, straighten, rotate, or reposition ANY element
- Do NOT redraw or re-render any part -- treat it as a mask/cutout operation
- If any text is blurry or partially obscured, leave it blurry/obscured
- If any part is cut off or not visible, leave it missing -- do NOT fill it in

Think of this as CUTTING OUT the product with scissors and placing it on white paper.
Nothing about the product itself should change.

Center the single product. Photorealistic, not illustrated.
"""


# --------------------------------------------------------------------------
# Scoring & cropping
# --------------------------------------------------------------------------

def score_photo(jpeg_bytes, item_name, openai_key):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    brightness = get_brightness(img)
    if brightness < MIN_BRIGHTNESS:
        return {"score": 1, "item_visible": False, "usable": False,
                "reason": f"Too dark (brightness={brightness:.0f})", "brightness": brightness}
    try:
        prompt = SCORE_PROMPT.format(item_name=item_name)
        raw = _openai_vision(prompt, [jpeg_bytes], openai_key, max_tokens=200, detail="high")
        result = _parse_json(raw)
        if result:
            result["brightness"] = brightness
            return result
    except Exception:
        pass
    return {"score": 0, "item_visible": False, "usable": False,
            "reason": "scoring error", "brightness": brightness}


def crop_from_pct(img, x1, y1, x2, y2, pad=0.02):
    w, h = img.size
    return img.crop((
        max(0, int((x1 - pad * 100) / 100 * w)),
        max(0, int((y1 - pad * 100) / 100 * h)),
        min(w, int((x2 + pad * 100) / 100 * w)),
        min(h, int((y2 + pad * 100) / 100 * h)),
    ))


def two_pass_crop(jpeg_bytes, item_name, openai_key):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    logger.info(f"    [CROP] Pass 1: Locating item on shelf...")
    raw = _openai_vision(
        LOCATE_COARSE_PROMPT.format(item_name=item_name),
        [_img_to_jpeg(img)], openai_key, max_tokens=200, detail="high"
    )
    coarse = _parse_json(raw)
    if not coarse or not coarse.get("found"):
        logger.info(f"    [CROP] Pass 1: Not found")
        return None
    if coarse.get("confidence") == "low":
        logger.info(f"    [CROP] Pass 1: Low confidence -- skipping")
        return None
    logger.info(f"    [CROP] Pass 1: Found ({coarse.get('confidence')}) -- zooming in...")
    zoomed = crop_from_pct(
        img, coarse["x1_pct"], coarse["y1_pct"],
        coarse["x2_pct"], coarse["y2_pct"], pad=0.08
    )
    raw2 = _openai_vision(
        LOCATE_PRECISE_PROMPT.format(item_name=item_name),
        [_img_to_jpeg(zoomed)], openai_key, max_tokens=200, detail="high"
    )
    precise = _parse_json(raw2)
    if not precise or not precise.get("found"):
        logger.info(f"    [CROP] Pass 2: Failed -- using coarse crop")
        return zoomed
    logger.info(f"    [CROP] Pass 2: Tight crop ({precise.get('confidence')})")
    return crop_from_pct(
        zoomed, precise["x1_pct"], precise["y1_pct"],
        precise["x2_pct"], precise["y2_pct"], pad=0.04
    )


def make_catalog_ready(img, target_size=1200):
    brightness = get_brightness(img)
    if 60 <= brightness < 100:
        img = ImageEnhance.Contrast(img).enhance(1.2)
        img = ImageEnhance.Brightness(img).enhance(1.15)
        logger.info(f"    [CATALOG] Gentle boost (was {brightness:.0f})")
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    img_rgb = img.convert("RGB")
    img_rgb.thumbnail((target_size - 100, target_size - 100), Image.LANCZOS)
    canvas.paste(img_rgb, (
        (target_size - img_rgb.width) // 2,
        (target_size - img_rgb.height) // 2,
    ))
    return canvas


def detect_and_fix_orientation(crop_img, openai_key):
    jpeg_bytes = _img_to_jpeg(crop_img, max_side=768)
    try:
        raw = _openai_vision(ORIENTATION_CHECK_PROMPT, [jpeg_bytes], openai_key,
                             max_tokens=100, detail="low")
        result = _parse_json(raw)
        if not result:
            return crop_img
        orientation = result.get('orientation', 'correct')
        confidence = result.get('confidence', 'low')
        if orientation == 'correct':
            logger.info(f"    [ORIENT] Orientation correct ({confidence})")
            return crop_img
        elif orientation == 'upside_down' and confidence != 'low':
            logger.info(f"    [ORIENT] Upside down detected ({confidence}) -- rotating 180")
            return crop_img.rotate(180, expand=True)
        elif orientation == 'rotated_90_cw' and confidence != 'low':
            logger.info(f"    [ORIENT] Rotated 90 CW ({confidence}) -- rotating 270")
            return crop_img.rotate(270, expand=True)
        elif orientation == 'rotated_90_ccw' and confidence != 'low':
            logger.info(f"    [ORIENT] Rotated 90 CCW ({confidence}) -- rotating 90")
            return crop_img.rotate(90, expand=True)
        else:
            logger.info(f"    [ORIENT] Low confidence ({orientation}) -- keeping as-is")
            return crop_img
    except Exception as e:
        logger.debug(f"    [ORIENT] Detection failed: {e}")
        return crop_img


def check_crop_quality(crop_img, openai_key):
    jpeg_bytes = _img_to_jpeg(crop_img, max_side=768)
    try:
        raw = _openai_vision(CROP_QUALITY_PROMPT, [jpeg_bytes], openai_key,
                             max_tokens=100, detail="low")
        result = _parse_json(raw)
        if result:
            usable = result.get('usable', True)
            reason = result.get('reason', '')
            if usable:
                logger.info(f"    [QUALITY] Crop OK -- {reason}")
            else:
                logger.info(f"    [QUALITY] Crop FAILED -- {reason}")
            return usable
    except Exception as e:
        logger.debug(f"    [QUALITY] Check failed: {e}")
    return True


# --------------------------------------------------------------------------
# AI background removal via gpt-image-1
# --------------------------------------------------------------------------

def ai_background_removal(source_jpeg_bytes, item_name, openai_key):
    """Remove background from product photo using gpt-image-1."""
    logger.info(f"    [AI_GEN] Background removal with gpt-image-1...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        img = Image.open(io.BytesIO(source_jpeg_bytes)).convert('RGBA')
        max_side = max(img.width, img.height)
        square = Image.new("RGBA", (max_side, max_side), (255, 255, 255, 255))
        square.paste(img, ((max_side - img.width) // 2, (max_side - img.height) // 2))
        square.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        square.save(buf, format="PNG")
        buf.seek(0)
        buf.name = "source.png"
        prompt = GPT_IMAGE_EDIT_PROMPT.format(item_name=item_name)
        response = client.images.edit(
            model="gpt-image-1", image=buf, prompt=prompt, size="1024x1024"
        )
        if response.data and response.data[0].b64_json:
            img_data = base64.b64decode(response.data[0].b64_json)
            enhanced = Image.open(io.BytesIO(img_data)).convert("RGB")
            logger.info(f"    [AI_GEN] Success ({enhanced.size[0]}x{enhanced.size[1]})")
            return enhanced
        elif response.data and response.data[0].url:
            img_resp = requests.get(response.data[0].url, timeout=30)
            if img_resp.status_code == 200:
                enhanced = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                logger.info(f"    [AI_GEN] Success ({enhanced.size[0]}x{enhanced.size[1]})")
                return enhanced
    except Exception as e:
        logger.info(f"    [AI_GEN] Failed: {e}")
    return None


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_items(filepath):
    """Load Easter items CSV. Uses ITEM_ID as unique key since each row is
    a unique MSID, and multiple MSIDs can share a DD_SIC."""
    items = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items()}
            item_id = norm.get('ITEM_ID', '').strip()
            if not item_id:
                continue
            vol_raw = norm.get('ITEM_VOLUME', '0')
            try:
                item_volume = int(float(vol_raw)) if vol_raw and vol_raw.lower() != 'null' else 0
            except (ValueError, TypeError):
                item_volume = 0
            pick_raw = norm.get('AVG_PICK_TIME_MIN', '')
            try:
                pick_time = float(pick_raw) if pick_raw and pick_raw.lower() != 'null' else 0.0
            except (ValueError, TypeError):
                pick_time = 0.0
            items[item_id] = {
                'item_id': item_id,
                'msid': norm.get('MSID', '').strip(),
                'dd_sic': norm.get('DD_SIC', '').strip(),
                'item_name': norm.get('ITEM_NAME', '').strip(),
                'business_name': norm.get('BUSINESS_NAME', '').strip(),
                'category': norm.get('CATEGORY', '').strip(),
                'has_image': norm.get('HAS_IMAGE', '').strip(),
                'photo_url': norm.get('PHOTO_URL', '').strip(),
                'item_volume': item_volume,
                'avg_pick_time': pick_time,
            }
    logger.info(f"Loaded {len(items)} items from {filepath}")
    has_img = sum(1 for v in items.values() if v['has_image'].lower() == 'yes')
    logger.info(f"  {has_img} with existing images, {len(items) - has_img} without")
    return items


def load_jets_data(filepath):
    """Load JETS photos CSV, keyed by DD_SIC."""
    if not filepath or not os.path.exists(filepath):
        logger.info("No JETS photos file provided")
        return {}
    groups = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = {k.strip().strip('"').upper(): v.strip().strip('"') for k, v in row.items()}
            dd_sic = (norm.get('DD_SIC_V2') or norm.get('DD_SIC') or '').strip()
            if not dd_sic:
                continue
            groups[dd_sic].append(norm)
    for dd_sic in groups:
        groups[dd_sic].sort(key=lambda r: int(r.get('RN', 0) or 0))
    logger.info(f"Loaded {sum(len(v) for v in groups.values())} JETS rows "
                f"for {len(groups)} DD_SICs")
    return groups


def load_community_data(filepaths):
    """Load community photos from one or more CSVs.
    Returns two dicts: by_dd_sic and by_msid, each mapping to lists of photo records.
    Deduplicates by PHOTO_UUID."""
    by_dd_sic = defaultdict(list)
    by_msid = defaultdict(list)
    seen_uuids = set()

    for filepath in filepaths:
        if not filepath or not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm = {k.strip().strip('"').upper(): v.strip().strip('"') for k, v in row.items()}
                photo_uuid = norm.get('PHOTO_UUID', '').strip()
                if photo_uuid in seen_uuids:
                    continue
                seen_uuids.add(photo_uuid)

                url = (norm.get('COMMUNITY_PHOTO_URL') or norm.get('IMAGE_URL')
                       or norm.get('PHOTO_URL') or norm.get('URL') or '').strip()
                if not url:
                    continue

                record = {
                    'url': url,
                    'photo_type': norm.get('PHOTO_TYPE', ''),
                    'store_id': norm.get('PHOTO_STORE_ID', ''),
                    'rn': norm.get('RN', '0'),
                }

                dd_sic = (norm.get('DD_SIC_V2') or norm.get('DD_SIC') or '').strip()
                msid = norm.get('MSID', '').strip()

                if dd_sic:
                    by_dd_sic[dd_sic].append(record)
                if msid:
                    by_msid[msid].append(record)

    # Sort each group by RN
    for key in by_dd_sic:
        by_dd_sic[key].sort(key=lambda r: int(r.get('rn', 0) or 0))
    for key in by_msid:
        by_msid[key].sort(key=lambda r: int(r.get('rn', 0) or 0))

    total_ddsic = sum(len(v) for v in by_dd_sic.values())
    total_msid = sum(len(v) for v in by_msid.values())
    logger.info(f"Loaded community photos: {len(seen_uuids)} unique photos")
    logger.info(f"  By DD_SIC: {total_ddsic} records for {len(by_dd_sic)} DD_SICs")
    logger.info(f"  By MSID:   {total_msid} records for {len(by_msid)} MSIDs")
    return by_dd_sic, by_msid


# --------------------------------------------------------------------------
# JETS processing (with early stop)
# --------------------------------------------------------------------------

def process_jets(dd_sic, jets_rows, item_name, openai_key, max_jets=25):
    """Try JETS zip photos for this item. Returns list of scored candidates."""
    scored, tried = [], 0
    for row in jets_rows[:max_jets]:
        zip_url = (row.get('PRODUCT_ZIP_URL') or '').strip()
        shelf_tag_id = (row.get('SHELF_TAG_ID') or '').strip()
        associations = (row.get('TAG_PRODUCT_ASSOCIATIONS') or '').strip()
        if not zip_url or not shelf_tag_id:
            continue
        product_filename = get_product_filename(shelf_tag_id, associations)
        if not product_filename:
            continue
        tried += 1
        logger.info(f"    [JETS] Candidate {tried}: {product_filename}")
        raw_bytes = download_and_extract_from_zip(zip_url, product_filename)
        if not raw_bytes:
            continue
        jpeg_bytes = convert_heif_to_jpeg(raw_bytes)
        if not jpeg_bytes:
            continue
        score_result = score_photo(jpeg_bytes, item_name, openai_key)
        score_val = score_result.get('score', 0)
        logger.info(f"    [JETS] Score: {score_val}/10 -- {score_result.get('reason', '')}")
        if score_val >= MIN_SCORE_THRESHOLD and score_result.get('usable'):
            scored.append({'jpeg_bytes': jpeg_bytes, 'url': zip_url,
                           'score': score_val, 'source': 'jets'})
            if score_val >= EARLY_STOP_SCORE:
                logger.info(f"    [JETS] Score {score_val} >= {EARLY_STOP_SCORE} -- stopping early")
                break
        time.sleep(0.3)
    return scored, tried


# --------------------------------------------------------------------------
# Community photo processing (with early stop)
# --------------------------------------------------------------------------

def process_community_photos(photos, item_name, openai_key, max_candidates=15):
    """Score community photos for this item. Returns list of scored candidates."""
    scored, tried = [], 0
    for photo in photos[:max_candidates]:
        url = photo['url']
        tried += 1
        logger.info(f"    [COMMUNITY] Candidate {tried}: {url[:80]}...")

        # ITEM_LOCATION photos can be used directly (already cropped closeups)
        if photo.get('photo_type') == 'PHOTO_TYPE_ITEM_LOCATION':
            jpeg_bytes = download_image_as_jpeg(url)
            if not jpeg_bytes:
                continue
            prompt = ITEM_LOCATION_CHECK_PROMPT.format(item_name=item_name)
            raw = _openai_vision(prompt, [jpeg_bytes], openai_key, max_tokens=200, detail="high")
            result = _parse_json(raw)
            if result and result.get('suitable') and result.get('confidence') != 'low':
                logger.info(f"    [COMMUNITY] Suitable item location photo!")
                scored.append({'jpeg_bytes': jpeg_bytes, 'url': url, 'score': 10,
                               'source': 'community', 'photo_type': 'item_location'})
                break  # score 10 = early stop
            continue

        jpeg_bytes = download_image_as_jpeg(url)
        if not jpeg_bytes:
            continue
        score_result = score_photo(jpeg_bytes, item_name, openai_key)
        score_val = score_result.get('score', 0)
        logger.info(f"    [COMMUNITY] Score: {score_val}/10 -- {score_result.get('reason', '')}")
        if score_val >= MIN_SCORE_THRESHOLD and score_result.get('usable'):
            scored.append({'jpeg_bytes': jpeg_bytes, 'url': url,
                           'score': score_val, 'source': 'community'})
            if score_val >= EARLY_STOP_SCORE:
                logger.info(f"    [COMMUNITY] Score {score_val} >= {EARLY_STOP_SCORE} -- stopping early")
                break
        time.sleep(0.3)
    return scored, tried


# --------------------------------------------------------------------------
# Main item processing
# --------------------------------------------------------------------------

def process_item(item_info, jets_rows, community_photos,
                 openai_key, serpapi_key, output_img_dir, args):
    """Process a single item: find best photo, crop, AI edit, save."""
    item_id = item_info['item_id']
    item_name = item_info['item_name']
    display = f"{item_info['business_name']} -- {item_name}" if item_info['business_name'] else item_name

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {item_id}: {display}")
    logger.info(f"  DD_SIC: {item_info['dd_sic']} | MSID: {item_info['msid']}")
    logger.info(f"  Volume: {item_info['item_volume']} | Category: {item_info['category']}")
    logger.info(f"  Has image: {item_info['has_image']} | "
                f"JETS rows: {len(jets_rows)} | Community photos: {len(community_photos)}")

    result = {
        'item_id': item_id,
        'msid': item_info['msid'],
        'dd_sic': item_info['dd_sic'],
        'item_name': item_name,
        'business_name': item_info['business_name'],
        'category': item_info['category'],
        'has_image': item_info['has_image'],
        'existing_photo_url': item_info['photo_url'],
        'item_volume': item_info['item_volume'],
        'avg_pick_time': item_info['avg_pick_time'],
    }

    if not jets_rows and not community_photos:
        logger.info(f"  No JETS or community photos available -- skipping")
        result['outcome'] = 'no_photos_available'
        return result

    # -- Step 1: Collect and score all candidates -------------------------
    all_scored = []

    # Try JETS first
    if jets_rows and not args.skip_jets:
        logger.info(f"  [JETS] {len(jets_rows)} photos available")
        jets_scored, jets_tried = process_jets(
            item_info['dd_sic'], jets_rows, item_name, openai_key, args.max_jets)
        all_scored.extend(jets_scored)
        result['jets_tried'] = jets_tried
        result['jets_scored'] = len(jets_scored)

    # Try community photos
    if community_photos and not args.skip_community:
        logger.info(f"  [COMMUNITY] {len(community_photos)} photos available")
        comm_scored, comm_tried = process_community_photos(
            community_photos, item_name, openai_key, args.max_community)
        all_scored.extend(comm_scored)
        result['community_tried'] = comm_tried
        result['community_scored'] = len(comm_scored)

    if not all_scored:
        logger.info(f"  No usable photos found")
        result['outcome'] = 'no_usable_photos'
        return result

    # Sort by score descending
    all_scored.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"  {len(all_scored)} scored candidate(s), best score={all_scored[0]['score']}/10 "
                f"from {all_scored[0]['source']}")

    # -- Step 2: Crop the best candidate ----------------------------------
    crop = None
    used = None
    for i, candidate in enumerate(all_scored):
        logger.info(f"  Trying crop on candidate {i+1} (score={candidate['score']}, "
                    f"source={candidate['source']})...")
        if candidate.get('photo_type') == 'item_location':
            crop = Image.open(io.BytesIO(candidate['jpeg_bytes'])).convert('RGB')
            used = candidate
            logger.info(f"  Item location photo -- using directly")
            break
        cropped = two_pass_crop(candidate['jpeg_bytes'], item_name, openai_key)
        if cropped is not None:
            crop = cropped
            used = candidate
            logger.info(f"  Crop succeeded on candidate {i+1}")
            break
        else:
            logger.info(f"  Crop failed on candidate {i+1}, trying next...")

    if crop is None:
        logger.info(f"  All crops failed -- using best photo uncropped")
        crop = Image.open(io.BytesIO(all_scored[0]['jpeg_bytes'])).convert('RGB')
        used = all_scored[0]

    # -- Step 3: Orientation fix ------------------------------------------
    crop = detect_and_fix_orientation(crop, openai_key)

    # -- Step 4: Quality gate ---------------------------------------------
    if not check_crop_quality(crop, openai_key):
        logger.info(f"  Crop too blurry/unclear -- skipping AI edit")
        result['outcome'] = 'crop_too_blurry'
        result['url_used'] = used['url']
        return result

    # -- Step 5: AI background removal ------------------------------------
    cropped_bytes = _img_to_jpeg(crop)
    generated = ai_background_removal(cropped_bytes, item_name, openai_key)

    if generated is not None:
        # -- Step 6: Search Google for reference images & verify text ------
        if serpapi_key:
            ref_images = search_reference_images(item_name, serpapi_key)
            result['refs_found'] = len(ref_images)
            if ref_images:
                generated = final_text_check_and_blur(generated, ref_images, openai_key)
        else:
            result['refs_found'] = 0

        catalog = make_catalog_ready(generated)
        out = output_img_dir / f"{item_id.replace('/', '_')}.jpg"
        catalog.save(out, "JPEG", quality=95)
        result['outcome'] = 'fixed'
        result['output_method'] = 'ai_edited'
        result['output_source'] = used['source']
        result['output_score'] = used['score']
        result['output_path'] = str(out)
        result['url_used'] = used['url']
        logger.info(f"  ai_edited -> {out}")
        return result

    # Fallback: raw crop on white canvas
    logger.info(f"  AI edit failed -- using raw crop as last resort")
    catalog = make_catalog_ready(crop)
    out = output_img_dir / f"{item_id.replace('/', '_')}.jpg"
    catalog.save(out, "JPEG", quality=95)
    result['outcome'] = 'fixed'
    result['output_method'] = 'raw_crop_fallback'
    result['output_source'] = used['source']
    result['output_score'] = used['score']
    result['output_path'] = str(out)
    result['url_used'] = used['url']
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Easter Item Checker v1.0')
    parser.add_argument('--items', required=True,
                        help='Items CSV (ITEM_ID, MSID, DD_SIC, ITEM_NAME, etc.)')
    parser.add_argument('--jets-photos', default=None,
                        help='JETS photos CSV (dd_sic + product zips)')
    parser.add_argument('--community-photos', nargs='+', default=[],
                        help='One or more community photos CSVs (MSID and/or DD_SIC based)')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--openai-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--serpapi-key', default=os.environ.get('SERPAPI_KEY'),
                        help='SerpAPI key for Google reference image search (or set SERPAPI_KEY)')
    parser.add_argument('--imgbb-key', default=os.environ.get('IMGBB_API_KEY'))
    parser.add_argument('--max-jets', type=int, default=10,
                        help='Max JETS photos to try per item')
    parser.add_argument('--max-community', type=int, default=10,
                        help='Max community photos to try per item')
    parser.add_argument('--max-items', type=int, default=0,
                        help='Max items to process (0 = all)')
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip first N items')
    parser.add_argument('--skip-jets', action='store_true')
    parser.add_argument('--skip-community', action='store_true')
    parser.add_argument('--sort-by', default='item_volume',
                        choices=['item_volume', 'avg_pick_time'],
                        help='Sort items by this field (descending)')
    args = parser.parse_args()

    if not args.openai_key:
        logger.error("OpenAI API key required. Set OPENAI_API_KEY or use --openai-key.")
        sys.exit(1)
    if not args.serpapi_key:
        logger.warning("No SerpAPI key -- text verification against reference images disabled.")

    output_dir = Path(args.output_dir)
    output_img_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    items = load_items(args.items)
    jets_data = load_jets_data(args.jets_photos)
    community_by_ddsic, community_by_msid = load_community_data(args.community_photos)

    # Sort by volume descending
    sorted_item_ids = sorted(
        items.keys(),
        key=lambda k: float(items[k].get('item_volume', 0)
                            if args.sort_by == 'item_volume'
                            else items[k].get('avg_pick_time', 0)),
        reverse=True)

    if args.offset > 0:
        sorted_item_ids = sorted_item_ids[args.offset:]
        logger.info(f"Offset {args.offset} -- {len(sorted_item_ids)} items remaining")

    results = []
    stats = defaultdict(int)

    logger.info(f"\nProcessing {len(sorted_item_ids)} items")
    logger.info(f"JETS data: {len(jets_data)} DD_SICs | "
                f"Community by DD_SIC: {len(community_by_ddsic)} | "
                f"Community by MSID: {len(community_by_msid)}")
    logger.info(f"Score threshold: {MIN_SCORE_THRESHOLD}/10 | "
                f"Early stop: {EARLY_STOP_SCORE}/10")
    logger.info(f"{'='*60}")

    for item_id in sorted_item_ids:
        if args.max_items > 0 and len(results) >= args.max_items:
            logger.info(f"\nReached --max-items limit ({args.max_items}), stopping.")
            break

        item_info = items[item_id]
        dd_sic = item_info['dd_sic']
        msid = item_info['msid']

        # Collect JETS rows by DD_SIC
        jets_rows = jets_data.get(dd_sic, [])

        # Collect community photos by DD_SIC and MSID, merge and deduplicate
        comm_by_ddsic = community_by_ddsic.get(dd_sic, [])
        comm_by_msid = community_by_msid.get(msid, [])

        # Merge: DD_SIC photos first, then MSID photos (deduplicated by URL)
        seen_urls = set()
        merged_community = []
        for photo in comm_by_ddsic + comm_by_msid:
            if photo['url'] not in seen_urls:
                seen_urls.add(photo['url'])
                merged_community.append(photo)

        result = process_item(item_info, jets_rows, merged_community,
                              args.openai_key, args.serpapi_key, output_img_dir, args)

        # Upload to imgBB if fixed
        result['public_image_url'] = ''
        result['imgbb_delete_url'] = ''
        if result.get('outcome') == 'fixed' and result.get('output_path') and args.imgbb_key:
            public_url, delete_url = upload_to_imgbb(
                result['output_path'], args.imgbb_key, item_info['item_name'])
            if public_url:
                result['public_image_url'] = public_url
                result['imgbb_delete_url'] = delete_url
            time.sleep(0.5)

        results.append(result)
        stats[result.get('outcome', 'unknown')] += 1

    # -- Write results CSV ------------------------------------------------
    fieldnames = [
        'item_id', 'msid', 'dd_sic', 'item_name', 'business_name', 'category',
        'has_image', 'existing_photo_url', 'item_volume', 'avg_pick_time',
        'outcome', 'output_source', 'output_method', 'output_score', 'output_path',
        'url_used', 'public_image_url', 'imgbb_delete_url',
        'refs_found', 'jets_tried', 'jets_scored', 'community_tried', 'community_scored',
    ]
    results_csv = output_dir / 'results.csv'
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    # -- Summary ----------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total items processed:          {len(results)}")
    logger.info(f"  Fixed (new image sourced):      {stats.get('fixed', 0)}")
    logger.info(f"  No photos available:             {stats.get('no_photos_available', 0)}")
    logger.info(f"  No usable photos:                {stats.get('no_usable_photos', 0)}")
    logger.info(f"  Crop too blurry:                 {stats.get('crop_too_blurry', 0)}")

    fixed_results = [r for r in results if r.get('outcome') == 'fixed']
    if fixed_results:
        method_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for r in fixed_results:
            method_counts[r.get('output_method', 'unknown')] += 1
            source_counts[r.get('output_source', 'unknown')] += 1
        logger.info(f"\n  Fixed by method:")
        for method, count in sorted(method_counts.items()):
            logger.info(f"    {method}: {count}")
        logger.info(f"  Fixed by source:")
        for source, count in sorted(source_counts.items()):
            logger.info(f"    {source}: {count}")

    fixed_volume = sum(r.get('item_volume', 0) for r in fixed_results)
    total_volume = sum(r.get('item_volume', 0) for r in results)
    if total_volume > 0:
        logger.info(f"\n  Volume coverage: {fixed_volume:,}/{total_volume:,} "
                     f"({100*fixed_volume/total_volume:.1f}%) of item orders")

    uploaded = sum(1 for r in results if r.get('public_image_url'))
    if uploaded > 0:
        logger.info(f"  Public URLs (imgBB):   {uploaded}/{stats.get('fixed', 0)}")

    logger.info(f"\n  Results: {results_csv}")
    logger.info(f"  Images:  {output_img_dir}/")


if __name__ == '__main__':
    main()
