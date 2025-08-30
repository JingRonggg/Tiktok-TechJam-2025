#!/usr/bin/env python3
# anonym.py — OWL-ViT-first anonymizer with Day/Night mode + quality gates
# - Ensemble of OWL models + multi-scale + mild augments
# - Day: green road-sign heuristic (HSV + morphology)
# - Night: LED panel heuristic (amber/green HSV + morphology)
# - NEW: area/aspect gates + edge-density “text-ish” gate to stop giant façades
# Outputs: <out>.jpg and <out>_overlay.jpg

import os; os.environ["TORCH_COMPILE_DISABLE"] = "1"
import argparse
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import pipeline as hfpipe

# ================= CONFIG =================
MAX_SIDE = 1800  # downscale long edge for speed (None to disable)

HF_MODELS = [
    "google/owlvit-base-patch32",
    "google/owlv2-base-patch16",
]

HF_PROMPTS = [
    "license plate","number plate","car plate",
    "bus destination sign","bus headsign","train headsign","arrival board","departure board",
    "street sign","road sign","traffic sign","direction sign","wayfinding sign",
    "airport sign","terminal sign","gate sign",
    "shop sign","store sign","storefront sign","building sign","building name",
    "billboard","advertising sign","neon sign","digital sign","electronic display","LED display",
]

HF_SCALES = [1.0, 1.6, 2.0]   # modest scales keep speed okay
HF_IOU_NMS = 0.50

HF_SCORE_THR_DAY   = 0.22
HF_SCORE_THR_NIGHT = 0.18

BRIGHT_AUGS_DAY   = ["orig", "desat0.85"]
BRIGHT_AUGS_NIGHT = ["orig", "gamma1.25", "clahe"]

# --------- NEW: OWL box quality gates ----------
BOX_MIN_FRAC   = 0.00008     # min box area as fraction of image
BOX_MAX_FRAC   = 0.12        # max area; suppress “whole building” (can raise to 0.18)
ASPECT_MIN     = 0.35        # too tall-narrow often railing posts
ASPECT_MAX     = 14.0        # super-long strips are often rail/roof trims
MIN_HEIGHT_FRAC = 0.015      # ignore ultra-thin bands (h < 1.5% of image height)

# Edge density “text-ish” gate (after adaptive-threshold)
EDGE_MIN = 0.010   # too smooth = likely plain wall/glass
EDGE_MAX = 0.300   # too busy/noisy = grid/trees
EDGE_KEEP_FOR = {"license plate","number plate","car plate",
                 "bus destination sign","bus headsign","train headsign",
                 "arrival board","departure board","digital sign","LED display"}  # labels allowed to bypass edge gate slightly

# ---- Day green-sign heuristic ----
GREEN_HSV_LOW  = [32, 35, 50]
GREEN_HSV_HIGH = [88,255,255]
MORPH_KERNEL   = (21, 7)
MORPH_ITER     = 3
GREEN_MIN_FRAC = 0.00012
GREEN_MAX_FRAC = 0.18
GREEN_ASPECT_MIN = 1.15

# ---- Night LED panel heuristic ----
LED_HSV_RANGES = [
    ((8, 80,120),(30,255,255)),   # amber/orange
    ((30,60,110),(85,255,255)),   # green
]
LED_KERNEL       = (17,5)
LED_ITER         = 2
LED_MIN_FRAC     = 0.00002   # a bit smaller to catch bus numbers
LED_MAX_FRAC     = 0.12
LED_ASPECT_MIN   = 1.05
LED_IOU_NMS      = 0.40

# --- Redaction ---
REDACTION_METHOD = "blur"   # "blur" | "pixelate"
BLUR_RADIUS      = 34
PIXELATE_BLOCK   = 30
PAD_PCT          = 0.06

SAVE_OVERLAY = True
VERBOSE = True
def log(*a):
    if VERBOSE: print(*a)

# ================ UTILS ===================
def pad_box(b, W, H, pct):
    x1,y1,x2,y2 = map(int,b); w,h = x2-x1, y2-y1; p = int(max(w,h)*pct)
    return [max(0,x1-p), max(0,y1-p), min(W,x2+p), min(H,y2+p)]

def nms(boxes, scores, thr):
    if not boxes: return []
    b = np.array(boxes, np.float32); s = np.array(scores, np.float32)
    order = s.argsort()[::-1]; keep=[]
    while order.size>0:
        i=int(order[0]); keep.append(i)
        xx1=np.maximum(b[i,0],b[order,0]); yy1=np.maximum(b[i,1],b[order,1])
        xx2=np.minimum(b[i,2],b[order,2]); yy2=np.minimum(b[i,3],b[order,3])
        w=np.maximum(0.0,xx2-xx1); h=np.maximum(0.0,yy2-yy1)
        inter=w*h
        area_i=(b[i,2]-b[i,0])*(b[i,3]-b[i,1]); area_o=(b[order,2]-b[order,0])*(b[order,3]-b[order,1])
        iouv= inter/(area_i+area_o-inter+1e-6)
        order = order[np.where(iouv<=thr)[0]]
    return keep

def draw_overlay(pil_img, owl_boxes, extra_boxes, final_boxes):
    vis = pil_img.copy(); d = ImageDraw.Draw(vis)
    for b in owl_boxes:   d.rectangle(b, outline=(0,200,0), width=3)    # green = OWL raw kept
    for b in extra_boxes: d.rectangle(b, outline=(0,128,255), width=3)  # blue  = heuristic
    for b in final_boxes: d.rectangle(b, outline=(255,0,0), width=4)    # red   = final
    return vis

def redact_roi(pil, b):
    x1,y1,x2,y2 = map(int,b)
    reg = pil.crop((x1,y1,x2,y2))
    if REDACTION_METHOD == "blur":
        pil.paste(reg.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)), (x1,y1))
    else:
        w,h = reg.size; pw,ph = max(1,w//PIXELATE_BLOCK), max(1,h//PIXELATE_BLOCK)
        pil.paste(reg.resize((pw,ph), Image.BILINEAR).resize((w,h), Image.NEAREST), (x1,y1))

def resize_max_side(pil_img, max_side):
    if not max_side: return pil_img, 1.0
    w,h = pil_img.size
    s = max(w,h)
    if s <= max_side: return pil_img, 1.0
    scale = max_side / float(s)
    new = pil_img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    return new, scale

# ---------- quick “text-ish” score ----------
def edge_density_ok(pil_img, box):
    """Return True if the crop looks panel/text-like (edge density within bounds)."""
    x1,y1,x2,y2 = map(int, box)
    crop = np.array(pil_img.crop((x1,y1,x2,y2)).convert("L"))
    if crop.size == 0: return False
    # speed: shrink large crops
    H,W = crop.shape
    m = max(H,W)
    if m > 512:
        s = 512.0 / m
        crop = cv2.resize(crop, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
    # mild contrast + edges
    crop = cv2.equalizeHist(crop)
    edges = cv2.Canny(crop, 80, 160)
    dens = edges.mean()/255.0  # 0..1
    return (EDGE_MIN <= dens <= EDGE_MAX)

# --------- Augmentations ----------
def apply_aug(pil_img, name):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if name.startswith("gamma"):
        g = float(name.replace("gamma",""))
        inv = 1.0 / g
        table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)
    elif name.startswith("desat"):
        s = float(name.replace("desat",""))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*s, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif name == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L,A,B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
        img = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def bright_augments(pil_img, mode):
    names = BRIGHT_AUGS_DAY if mode=="day" else BRIGHT_AUGS_NIGHT
    return [(n, apply_aug(pil_img, n)) for n in names]

# --------- Mode decide (day/night) ----------
def scene_mode(pil_img):
    hsv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV)
    v_med = np.median(hsv[:,:,2])
    mode = "night" if v_med <= 110 else "day"
    log(f"[MODE] median V={v_med:.1f} → {mode}")
    return mode

# ============== DETECTORS =================
def owl_boxes_ensemble(pil_img, mode):
    device = 0 if torch.cuda.is_available() else -1
    model_kwargs = {"torch_dtype": torch.float16} if device==0 else {}
    pipes = [hfpipe("zero-shot-object-detection", model=m, device=device, model_kwargs=model_kwargs)
             for m in HF_MODELS]

    thr = HF_SCORE_THR_NIGHT if mode=="night" else HF_SCORE_THR_DAY
    W0,H0 = pil_img.size
    img_area = W0*H0
    all_boxes, all_scores = [], []
    kept_debug = []

    for aug_name, aug_img in bright_augments(pil_img, mode):
        for s in HF_SCALES:
            im = aug_img if s==1.0 else aug_img.resize((int(W0*s), int(H0*s)), Image.BICUBIC)
            for p in pipes:
                res = p(im, candidate_labels=HF_PROMPTS)
                kept = 0
                for r in res:
                    sc = float(r["score"])
                    if sc < thr: 
                        continue
                    lbl = str(r.get("label","")).lower()
                    bx = [int(r["box"]["xmin"]/s), int(r["box"]["ymin"]/s),
                          int(r["box"]["xmax"]/s), int(r["box"]["ymax"]/s)]
                    x1,y1,x2,y2 = bx
                    w,h = max(0,x2-x1), max(0,y2-y1)
                    if w==0 or h==0: 
                        continue

                    # --- NEW: geometry & area guards ---
                    area_frac = (w*h) / float(img_area)
                    aspect = w / max(1.0, float(h))
                    if area_frac < BOX_MIN_FRAC: 
                        continue
                    if h < MIN_HEIGHT_FRAC * H0:
                        continue
                    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                        continue
                    # cap oversized boxes unless clearly a billboard/panel
                    is_panelish = any(k in lbl for k in ["billboard","board","display","sign","headsign"])
                    if area_frac > BOX_MAX_FRAC and not is_panelish:
                        continue

                    # --- NEW: edge-density text-ish gate ---
                    # allow slight bypass for plates/panels
                    bypass = any(k in lbl for k in EDGE_KEEP_FOR)
                    if not bypass and not edge_density_ok(pil_img, bx):
                        continue

                    all_boxes.append(bx); all_scores.append(sc); kept += 1
                    kept_debug.append((lbl, sc, area_frac, aspect))
                log(f"[OWL] aug={aug_name:<8} s={s:.2f} {p.model.name_or_path.split('/')[-1]} kept={kept}")

    if not all_boxes:
        return []

    keep = nms(all_boxes, all_scores, HF_IOU_NMS)
    out = [all_boxes[i] for i in keep]
    log(f"[OWL] total after NMS: {len(out)}")
    return out

def green_blade_boxes(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H,W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(GREEN_HSV_LOW,  np.uint8)
    hi = np.array(GREEN_HSV_HIGH, np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]; imgA=W*H
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        A=w*h
        if A < GREEN_MIN_FRAC*imgA or A > GREEN_MAX_FRAC*imgA: continue
        if (w/max(1.0,float(h))) < GREEN_ASPECT_MIN: continue
        boxes.append([x,y,x+w,y+h])
    if boxes:
        keep = nms(boxes, [1.0]*len(boxes), 0.30)
        boxes = [boxes[i] for i in keep]
    log(f"[GREEN] boxes: {len(boxes)}")
    return boxes

def led_panel_boxes(pil_img):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H,W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros((H,W), np.uint8)
    for lo,hi in LED_HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo,np.uint8), np.array(hi,np.uint8))
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LED_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=LED_ITER)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]; imgA=W*H
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c); A=w*h
        if A < LED_MIN_FRAC*imgA or A > LED_MAX_FRAC*imgA: continue
        if (w/max(1.0,float(h))) < LED_ASPECT_MIN: continue
        boxes.append([x,y,x+w,y+h])
    if boxes:
        keep = nms(boxes, [1.0]*len(boxes), LED_IOU_NMS)
        boxes = [boxes[i] for i in keep]
    log(f"[LED] boxes: {len(boxes)}")
    return boxes

# ================ MAIN ====================
def main(inp, outp):
    pil_full = Image.open(inp).convert("RGB")
    pil, scale = resize_max_side(pil_full, MAX_SIDE)
    W,H = pil.size

    mode = scene_mode(pil)
    owl = owl_boxes_ensemble(pil, mode)
    extras = green_blade_boxes(pil) if mode=="day" else led_panel_boxes(pil)

    all_boxes = owl + extras
    all_boxes = [pad_box(b, W, H, PAD_PCT) for b in all_boxes]
    if all_boxes:
        keep = nms(all_boxes, [1.0]*len(all_boxes), 0.40)
        all_boxes = [all_boxes[i] for i in keep]

    # map back if downscaled
    if scale != 1.0:
        inv = 1.0/scale
        def up(b): return [int(b[0]*inv), int(b[1]*inv), int(b[2]*inv), int(b[3]*inv)]
        final_boxes = [up(b) for b in all_boxes]
        overlay_base = pil_full
    else:
        final_boxes = all_boxes
        overlay_base = pil

    if SAVE_OVERLAY:
        overlay = draw_overlay(overlay_base, final_boxes, [], final_boxes)
        overlay.save(str(Path(outp).with_suffix("")) + "_overlay.jpg", quality=95)

    out = overlay_base.copy()
    for b in final_boxes: redact_roi(out, b)
    out.save(outp, quality=95)
    log(f"[OK] redacted {len(final_boxes)} regions → {outp}")

# --------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    a = ap.parse_args()
    main(a.input, a.output)
