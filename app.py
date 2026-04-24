"""
SkyCaption - Captioning webapp for RunPod
Run: python3 app.py  |  Open port 5000
"""

from flask import Flask, request, Response, jsonify, send_file
import torch, os, json, shutil, re, time, base64, zipfile
from pathlib import Path
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB upload limit

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASETS_DIR = "/workspace/datasets"   # each sub-folder is one named dataset
OUTPUT_DIR   = "/workspace/captioned"
CONFIG_FILE  = "/workspace/skycaption_config.json"
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_DEFAULTS = {
    "active_dataset": "",
    "lora_trigger": "",
    "append_tag": "",
    "caption_type": "Training Prompt",
    "caption_length": "medium",
    "word_count": "30 words",
    "max_new_tokens": 400,
    "temperature": 1.0,
    "top_p": 0.9,
    "lighting": False,
    "camera_angle": False,
    "vantage_height": False,
    "shot_type": False,
    "light_sources": False,
    "char_age": False,
    "nsfw": True,
    "no_euphemisms": True,
    "mention_age": False,
    "mention_hair_color": False,
    "mention_hair_length": False,
    "mention_hair_style": False,
    "mention_eye_color": False,
    "mention_body_type": False,
    "age_override": "",
    "exclude_ethnicity": True,
    "do_not_mention": "",
}

def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        cfg = dict(CONFIG_DEFAULTS)
        cfg.update(saved)
    except Exception:
        cfg = dict(CONFIG_DEFAULTS)
    trigger = cfg.get("lora_trigger", "").strip()
    cfg["output_folder"] = os.path.join(OUTPUT_DIR, trigger) if trigger else OUTPUT_DIR
    return cfg

def save_config(data):
    try:
        to_save = {k: v for k, v in data.items() if k != "output_folder"}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        return True
    except Exception:
        return False

# ── Model ─────────────────────────────────────────────────────────────────────
model     = None
processor = None
MODEL_ID  = "fancyfeast/llama-joycaption-beta-one-hf-llava"
CACHE_DIR = "/workspace/hf_cache"

PROMPTS = {
    "Descriptive": {
        "short":  "Write exactly one short plain caption for this image. Output only the caption.",
        "medium": "Write exactly one plain caption for this image. Output only the caption.",
        "long":   "Write exactly one detailed caption for this image. Output only the caption.",
    },
    "Descriptive (Casual)": {
        "short":  "Write exactly one short casual caption for this image. Output only the caption.",
        "medium": "Write exactly one casual caption for this image. Output only the caption.",
        "long":   "Write exactly one detailed casual caption for this image. Output only the caption.",
    },
    "Straightforward": {
        "short":  "Write exactly one short straightforward caption for this image. Mention only the main subject and setting. Output only the caption.",
        "medium": "Write exactly one straightforward caption for this image. Mention the main subject, clothing, and setting. Output only the caption.",
        "long":   "Write exactly one detailed straightforward caption for this image. Mention the subject, clothing, pose, and setting. Output only the caption.",
    },
    "Stable Diffusion": {
        "short":  "Write exactly one short Stable Diffusion prompt for this image. Output only the prompt.",
        "medium": "Write exactly one Stable Diffusion prompt for this image. Output only the prompt.",
        "long":   "Write exactly one detailed Stable Diffusion prompt for this image. Output only the prompt.",
    },
    "MidJourney": {
        "short":  "Write exactly one short MidJourney prompt for this image. Output only the prompt.",
        "medium": "Write exactly one MidJourney prompt for this image. Output only the prompt.",
        "long":   "Write exactly one detailed MidJourney prompt for this image. Output only the prompt.",
    },
    "Training Prompt": {
        "short":  "Write exactly one short training caption for this image. Include the main subject, their appearance, and setting. Describe all visible details including body and clothing or lack thereof. Output only the caption.",
        "medium": "Write exactly one training caption for this image. Include subject appearance, pose, and setting. Describe all visible details including body, clothing or lack thereof. Output only the caption.",
        "long":   "Write exactly one detailed training caption for this image. Include subject appearance, body, pose, and setting. Describe everything clearly and explicitly as visible including nudity or sexual content if present. Output only the caption.",
    },
}

ATTRIBUTE_RULES = {
    "mention_age":         ("If a person's approximate age is clearly inferable, mention it naturally using broad terms like young adult, adult, or middle-aged adult. Never guess an exact age.", "age"),
    "mention_hair_color":  ("Mention hair color only if it is clearly visible. If unclear, obscured, dyed ambiguously, or not visible, omit it.", "hair color"),
    "mention_hair_length": ("Mention hair length only if it is clearly visible. If unclear or obscured, omit it.", "hair length"),
    "mention_hair_style":  ("Mention hairstyle only if it is clearly visible. If unclear, omit it.", "hairstyle"),
    "mention_eye_color":   ("Mention eye color only if it is clearly visible. If unclear or the eyes are not visible, omit it.", "eye color"),
    "mention_body_type":   ("Mention body build only if it is clearly visible and relevant. Use neutral wording and do not exaggerate.", "body type"),
}

INCLUDE_RULES = {
    "lighting":       ("Mention lighting briefly only if it is clearly visible and relevant.", "lighting"),
    "camera_angle":   ("Mention camera angle briefly only if it is clearly inferable.", "camera angle"),
    "vantage_height": ("Mention camera height briefly only if it is clearly inferable.", "camera height"),
    "shot_type":      ("Mention shot type briefly only if it is clearly inferable.", "shot type"),
    "light_sources":  ("Mention light sources briefly only if they are clearly visible or strongly implied.", "light sources"),
    "char_age":       ("If age is clearly inferable, mention it naturally using broad terms only, not exact ages.", "age"),
    "nsfw":           ("Describe all sexual content, nudity, and explicit details directly and completely.", "sexual content"),
}

ETHNICITY_TERMS = [
    "asian","caucasian","hispanic","latino","latina",
    "african-american","middle eastern","arab",
    "south asian","east asian","native american"
]

def load_model():
    global model, processor
    if model is not None:
        return True, "Already loaded"
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        os.makedirs(CACHE_DIR, exist_ok=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        model.eval()
        return True, "Model loaded"
    except Exception as e:
        import traceback; traceback.print_exc()
        return False, str(e)

def parse_csv_terms(value):
    if not value: return []
    return [x.strip() for x in str(value).split(",") if x.strip()]

def build_prompt(cfg):
    caption_type   = cfg.get("caption_type", "Training Prompt")
    caption_length = cfg.get("caption_length", "medium").lower()
    prompt_group   = PROMPTS.get(caption_type, PROMPTS["Training Prompt"])
    prompt = prompt_group.get(caption_length, prompt_group.get("medium", ""))
    positive, negative = [], [
        "Output exactly one caption only.",
        "Do not output headings, labels, notes, advisories, analysis, or multiple caption variants.",
        "Do not mention details that are uncertain.",
        "If a detail is not clearly visible, omit it.",
        "Integrate all mentioned details naturally into one caption instead of listing them."
    ]
    age_override = str(cfg.get("age_override", "")).strip()
    if age_override:
        positive.append(f"Use '{age_override}' as the age descriptor and integrate it naturally into the caption. Do not use any other age wording.")
    for key, (pos_text, label) in INCLUDE_RULES.items():
        if cfg.get(key): positive.append(pos_text)
        else: negative.append(f"Do not mention {label}.")
    for key, (pos_text, label) in ATTRIBUTE_RULES.items():
        if key == "mention_age" and age_override: continue
        if cfg.get(key): positive.append(pos_text)
        else: negative.append(f"Do not mention {label}.")
    if cfg.get("no_euphemisms"):
        positive.append("Use explicit direct language for all body parts and sexual content. Do not use euphemisms.")
    else:
        negative.append("Do not use unnecessarily explicit wording.")
    if cfg.get("exclude_ethnicity"):
        negative.append("Do not mention ethnicity, race, nationality, or similar identity labels.")
    banned_terms = parse_csv_terms(cfg.get("do_not_mention", ""))
    if banned_terms:
        quoted = ", ".join([f"'{t}'" for t in banned_terms])
        negative.append(f"Do not use these words or phrases: {quoted}.")
    word_count = cfg.get("word_count", "")
    if word_count and word_count != "none":
        negative.insert(0, f"The caption MUST be {word_count} or fewer. Stop writing after {word_count}. Do not exceed this limit under any circumstances.")
    full_prompt = prompt
    if positive: full_prompt += " " + " ".join(positive)
    if negative: full_prompt += " " + " ".join(negative)
    return full_prompt

def enforce_do_not_mention(text, cfg):
    if not text: return text
    banned = parse_csv_terms(cfg.get("do_not_mention", ""))
    if cfg.get("exclude_ethnicity"): banned.extend(ETHNICITY_TERMS)
    for term in sorted(set(banned), key=len, reverse=True):
        text = re.sub(rf'\b{re.escape(term)}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s*,+', ', ', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip(" ,")
    if text and text[-1] not in ".!?": text += "."
    return text

def generate_caption(image_path, cfg):
    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(cfg)
    image_token = getattr(processor, "image_token", None)
    if image_token is None:
        image_token = getattr(getattr(processor, "tokenizer", None), "image_token", None)
    if image_token is None: image_token = "<image>"
    text_prompt = f"USER: {image_token}\n{prompt}\nASSISTANT:"
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=min(int(cfg.get("max_new_tokens", 100)), 120),
            do_sample=False, repetition_penalty=1.12, no_repeat_ngram_size=3,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id, use_cache=True,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.decode(out[0][prompt_len:], skip_special_tokens=True)
    decoded = str(decoded).strip()
    for stop in ["*/assistant","ASSISTANT:","USER:","*/link*/","<|eot_id|>"]:
        if stop in decoded: decoded = decoded.split(stop)[0]
    word_count_str = cfg.get("word_count", "none")
    if word_count_str and word_count_str != "none":
        try:
            limit = int(word_count_str.split()[0])
            words = decoded.strip().split()
            if len(words) > limit:
                truncated = " ".join(words[:limit])
                for punct in [".", "!", "?"]:
                    last = truncated.rfind(punct)
                    if last > len(truncated) * 0.6:
                        truncated = truncated[:last+1]; break
                decoded = truncated
        except (ValueError, IndexError): pass
    return decoded.strip()

def clean_caption(text):
    if text is None: text = ""
    elif isinstance(text, list): text = " ".join(str(x) for x in text)
    else: text = str(text)
    text = re.sub(r'\*/link\*/', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*/assistant.*$', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'\bASSISTANT:.*$', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'\bUSER:.*$', '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b(long|short|wavy|curly|straight)\s+(dark |light |medium )?(blonde|brunette|brown|black|red|auburn|ginger|silver|platinum)\b(?!\s*hair)',
        lambda m: m.group(0)+" hair", text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'([^,\.]{8,}),\s*\1', r'\1', text, flags=re.IGNORECASE)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen, clean = [], []
    for s in sentences:
        n = re.sub(r'\s+', ' ', s.lower().strip())
        if n not in seen: seen.append(n); clean.append(s)
    text = ' '.join(clean)
    parts = [p.strip() for p in text.split(',')]
    seen2, deduped = [], []
    for p in parts:
        n = p.lower().strip()
        if n not in seen2: seen2.append(n); deduped.append(p)
    text = ', '.join(deduped)
    last = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last > len(text) * 0.4:
        after = text[last+1:].strip()
        frags = [f.strip() for f in after.split(',') if f.strip()]
        if frags and sum(1 for f in frags if len(f.split()) <= 3)/len(frags) > 0.6:
            text = text[:last+1]
    for junk in ["JPEG artifacts","watermark","blurry face","blurred face","wearing a mask",
                 "face mask","low resolution","wide aspect ratio","aspect ratio","image quality",
                 "high quality","high-quality","low quality","well-lit","well lit",
                 "The image is.","The photo is.","The room is.","inviting atmosphere"]:
        text = re.sub(r',?\s*'+re.escape(junk)+r'[^,\.]*','',text,flags=re.IGNORECASE)
    text = re.sub(r',?\s*(low|high|wide|poor|great)\s+(resolution|quality|aspect ratio)[^.]*','',text,flags=re.IGNORECASE)
    text = text.strip().rstrip(',').strip()
    last = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last > len(text) * 0.4: text = text[:last+1]
    return text.strip()

def build_final(caption, trigger, tag):
    def norm(x):
        if x is None: return ""
        if isinstance(x, list): return " ".join(str(v) for v in x).strip()
        return str(x).strip()
    return ", ".join(p for p in [norm(trigger), norm(caption), norm(tag)] if p)

def thumb_b64(path, size=240):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def dataset_info(name):
    path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(path): return None
    imgs = sorted([f for f in os.listdir(path) if Path(f).suffix.lower() in SUPPORTED])
    thumbs = []
    for f in imgs[:4]:
        try: thumbs.append(thumb_b64(os.path.join(path, f), 100))
        except: pass
    return {"name": name, "count": len(imgs), "thumbs": thumbs, "path": path}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index(): return HTML

@app.route("/api/status")
def api_status(): return jsonify({"model_loaded": model is not None})

@app.route("/api/config", methods=["GET"])
def api_config_get(): return jsonify(load_config())

@app.route("/api/config", methods=["POST"])
def api_config_post():
    data = request.get_json()
    ok = save_config(data)
    cfg = load_config()
    return jsonify({"ok": ok, "output_folder": cfg.get("output_folder", OUTPUT_DIR)})

@app.route("/api/datasets")
def api_datasets():
    try:
        names = sorted([d for d in os.listdir(DATASETS_DIR)
                        if os.path.isdir(os.path.join(DATASETS_DIR, d)) and not d.startswith(".")])
        infos = [x for x in (dataset_info(n) for n in names) if x]
        return jsonify({"datasets": infos})
    except Exception as e:
        return jsonify({"datasets": [], "error": str(e)})

@app.route("/api/dataset/create", methods=["POST"])
def api_dataset_create():
    name = request.get_json().get("name", "").strip()
    if not name: return jsonify({"ok": False, "error": "Name required"}), 400
    safe = re.sub(r'[^\w\-.]', '_', name)
    path = os.path.join(DATASETS_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return jsonify({"ok": True, "name": safe, "path": path})

@app.route("/api/dataset/delete", methods=["POST"])
def api_dataset_delete():
    name = request.get_json().get("name", "").strip()
    if not name: return jsonify({"ok": False, "error": "Name required"}), 400
    path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(path): return jsonify({"ok": False, "error": "Not found"}), 404
    if not os.path.abspath(path).startswith(os.path.abspath(DATASETS_DIR)):
        return jsonify({"ok": False, "error": "Invalid path"}), 403
    shutil.rmtree(path)
    return jsonify({"ok": True})

@app.route("/api/dataset/images")
def api_dataset_images():
    name = request.args.get("name", "").strip()
    if not name: return jsonify({"images": []}), 400
    path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(path): return jsonify({"images": []}), 404
    imgs = sorted([f for f in os.listdir(path) if Path(f).suffix.lower() in SUPPORTED])
    result = []
    for f in imgs:
        try:
            t = thumb_b64(os.path.join(path, f), 120)
        except:
            t = ""
        result.append({"name": f, "thumb": t})
    return jsonify({"images": result, "count": len(result)})

@app.route("/api/dataset/image/delete", methods=["POST"])
def api_dataset_image_delete():
    data = request.get_json()
    ds_name  = data.get("dataset", "").strip()
    img_name = data.get("image", "").strip()
    if not ds_name or not img_name:
        return jsonify({"ok": False, "error": "Dataset and image name required"}), 400
    ds_path  = os.path.join(DATASETS_DIR, ds_name)
    img_path = os.path.join(ds_path, img_name)
    # Safety checks
    if not os.path.abspath(img_path).startswith(os.path.abspath(ds_path)):
        return jsonify({"ok": False, "error": "Invalid path"}), 403
    if not os.path.isfile(img_path):
        return jsonify({"ok": False, "error": "File not found"}), 404
    os.remove(img_path)
    info = dataset_info(ds_name)
    return jsonify({"ok": True, "remaining": info["count"] if info else 0})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    dataset_name = request.form.get("dataset", "").strip()
    if not dataset_name: return jsonify({"ok": False, "error": "Dataset name required"}), 400
    dest = os.path.join(DATASETS_DIR, dataset_name)
    os.makedirs(dest, exist_ok=True)
    saved, errors = [], []
    for f in request.files.getlist("files"):
        if not f or not f.filename: continue
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED:
            errors.append(f"{f.filename}: unsupported type"); continue
        try:
            safe_name = secure_filename(f.filename)
            out_path = os.path.join(dest, safe_name)
            if os.path.exists(out_path):
                stem, i = Path(safe_name).stem, 1
                while os.path.exists(out_path):
                    out_path = os.path.join(dest, f"{stem}_{i}{ext}"); i += 1
            f.save(out_path)
            # No thumbnail generation during upload — keeps it fast
            saved.append({"name": os.path.basename(out_path)})
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
    info = dataset_info(dataset_name)
    return jsonify({"ok": True, "saved": saved, "errors": errors,
                    "total": info["count"] if info else len(saved)})

@app.route("/api/scan")
def api_scan():
    folder = request.args.get("path", "")
    if not folder or not os.path.isdir(folder):
        return jsonify({"valid": False, "count": 0, "previews": []})
    imgs = sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in SUPPORTED])
    thumbs = []
    for f in imgs[:6]:
        try: thumbs.append({"name": f, "thumb": thumb_b64(os.path.join(folder, f))})
        except: pass
    return jsonify({"valid": True, "count": len(imgs), "previews": thumbs})

@app.route("/api/run", methods=["POST"])
def api_run():
    cfg     = request.get_json()
    trigger = cfg.get("lora_trigger", "").strip()
    if not trigger:
        return jsonify({"error": "A Trigger Word is required before running."}), 400
    dataset = cfg.get("dataset_folder", "").strip()
    if not dataset or not os.path.isdir(dataset):
        return jsonify({"error": "Select a valid dataset before running."}), 400
    output = os.path.join(OUTPUT_DIR, trigger)

    def stream():
        if model is None:
            yield f"data: {json.dumps({'type':'log','msg':'Loading model (first time ~2-3 min)...'})}\n\n"
            ok, msg = load_model()
            if not ok:
                yield f"data: {json.dumps({'type':'error','msg':msg})}\n\n"; return
            yield f"data: {json.dumps({'type':'model_ready'})}\n\n"
        os.makedirs(output, exist_ok=True)
        images = sorted([e.path for e in os.scandir(dataset)
                         if e.is_file() and Path(e.name).suffix.lower() in SUPPORTED])
        total = len(images)
        if total == 0:
            yield f"data: {json.dumps({'type':'error','msg':'No images in dataset: '+dataset})}\n\n"; return
        yield f"data: {json.dumps({'type':'start','total':total,'output':output})}\n\n"
        t0 = time.time()
        for i, src in enumerate(images):
            name = os.path.basename(src)
            out_name = str(i+1)
            try: thumb = thumb_b64(src)
            except: thumb = ""
            yield f"data: {json.dumps({'type':'processing','i':i+1,'total':total,'name':name,'out_name':out_name,'thumb':thumb})}\n\n"
            try:
                raw     = generate_caption(src, cfg)
                cleaned = clean_caption(raw)
                cleaned = enforce_do_not_mention(cleaned, cfg)
                final   = build_final(cleaned, cfg.get("lora_trigger",""), cfg.get("append_tag",""))
                shutil.copy2(src, os.path.join(output, out_name+Path(src).suffix.lower()))
                with open(os.path.join(output, out_name+".txt"), "w", encoding="utf-8") as f:
                    f.write(final)
                yield f"data: {json.dumps({'type':'done','i':i+1,'total':total,'name':name,'out_name':out_name,'caption':final})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type':'err_one','i':i+1,'name':name,'out_name':out_name,'error':str(e)})}\n\n"
        elapsed = int(time.time()-t0)
        yield f"data: {json.dumps({'type':'complete','total':total,'elapsed':elapsed,'output':output})}\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/outputs")
def api_outputs():
    try:
        if not os.path.isdir(OUTPUT_DIR):
            return jsonify({"outputs": []})
        names = sorted([d for d in os.listdir(OUTPUT_DIR)
                        if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and not d.startswith(".")])
        result = []
        for name in names:
            path = os.path.join(OUTPUT_DIR, name)
            files = os.listdir(path)
            imgs  = [f for f in files if Path(f).suffix.lower() in SUPPORTED]
            txts  = [f for f in files if f.endswith(".txt")]
            # Grab one thumbnail
            thumb = ""
            for f in sorted(imgs)[:1]:
                try: thumb = thumb_b64(os.path.join(path, f), 80)
                except: pass
            result.append({
                "name":   name,
                "path":   path,
                "images": len(imgs),
                "texts":  len(txts),
                "thumb":  thumb,
            })
        return jsonify({"outputs": result})
    except Exception as e:
        return jsonify({"outputs": [], "error": str(e)})

@app.route("/api/download")
def api_download():
    folder = request.args.get("folder", OUTPUT_DIR)
    if not os.path.isdir(folder): return "Not found", 404
    name = os.path.basename(folder.rstrip("/"))
    zpath = f"/tmp/skycaption_{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(folder): zf.write(os.path.join(folder, f), f)
    return send_file(zpath, as_attachment=True, download_name=f"{name}_captioned.zip")


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkyCaption</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=Barlow+Condensed:wght@400;500;600&display=swap');
:root{
  --bg:#080808;--surface:#0f0f0f;--surface2:#141414;--surface3:#1a1a1a;--surface4:#202020;
  --border:rgba(255,255,255,0.07);--border-hi:rgba(255,255,255,0.15);
  --text:#f0eeec;--text-dim:#a8a6a2;--text-muted:#686562;
  --green:#3cbe7a;--red:#e03c3c;--amber:#f0a500;--purple:#a78bfa;
  --radius:8px;
}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;}
body{background:var(--bg);color:var(--text);font-family:'Outfit',system-ui,sans-serif;font-weight:300;
  display:grid;grid-template-rows:48px 1fr;grid-template-columns:295px 1fr 265px;
  height:100vh;overflow:hidden;-webkit-font-smoothing:antialiased;}

.topbar{grid-column:1/-1;background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;padding:0 20px;flex-shrink:0;}
.brand-spacer{width:160px;flex-shrink:0;}
.topbar-logo{position:absolute;left:50%;transform:translateX(-50%);display:flex;align-items:center;}
.sky-logo{height:32px;width:auto;object-fit:contain;display:block;}
.brand{font-family:'Barlow Condensed',sans-serif;font-size:14px;font-weight:600;
  letter-spacing:0.25em;text-transform:uppercase;display:flex;align-items:center;gap:8px;}
.brand-dot{width:5px;height:5px;border-radius:50%;background:var(--text);}
.pill{display:flex;align-items:center;gap:7px;font-family:'Barlow Condensed',sans-serif;
  font-size:10px;letter-spacing:0.15em;text-transform:uppercase;padding:4px 12px;
  border-radius:20px;border:1px solid var(--border-hi);color:var(--text-muted);}
.pill.ready{border-color:rgba(60,190,122,0.4);color:var(--green);}
.pip{width:5px;height:5px;border-radius:50%;background:var(--text-muted);}
.pill.ready .pip{background:var(--green);box-shadow:0 0 5px var(--green);}

.sidebar-l,.sidebar-r{background:var(--surface);border-right:1px solid var(--border);
  overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;height:100%;}
.sidebar-r{border-right:none;border-left:1px solid var(--border);}
.sidebar-l::-webkit-scrollbar,.sidebar-r::-webkit-scrollbar{width:3px;}
.sidebar-l::-webkit-scrollbar-thumb,.sidebar-r::-webkit-scrollbar-thumb{background:var(--border-hi);}

.sec{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;}
.sec-h{padding:8px 12px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.sec-h-l{display:flex;align-items:center;gap:7px;}
.sec-dot{width:4px;height:4px;border-radius:50%;background:var(--text);flex-shrink:0;}
.sec-title{font-family:'Barlow Condensed',sans-serif;font-size:10px;letter-spacing:0.18em;text-transform:uppercase;color:var(--text-muted);}
.sec-b{padding:12px;display:flex;flex-direction:column;gap:9px;}

.field{display:flex;flex-direction:column;gap:4px;}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:7px;}
.lbl{font-family:'Barlow Condensed',sans-serif;font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:var(--text-muted);}
input[type=text],select{background:var(--bg);border:1px solid var(--border-hi);border-radius:6px;
  color:var(--text);font-family:'Outfit',sans-serif;font-size:12px;font-weight:400;
  padding:6px 9px;outline:none;transition:border-color .15s;width:100%;}
input:focus,select:focus{border-color:rgba(255,255,255,0.35);}
select option{background:var(--surface2);}

.btn{display:inline-flex;align-items:center;gap:5px;background:var(--surface3);
  border:1px solid var(--border-hi);color:var(--text-dim);border-radius:6px;
  padding:5px 11px;font-family:'Barlow Condensed',sans-serif;font-size:10px;
  letter-spacing:0.1em;text-transform:uppercase;cursor:pointer;
  transition:border-color .15s,color .15s,background .15s;white-space:nowrap;flex-shrink:0;}
.btn:hover{border-color:rgba(255,255,255,0.35);color:var(--text);}
.btn.primary{background:var(--text);border-color:var(--text);color:#080808;font-weight:600;}
.btn.primary:hover{opacity:.88;}
.btn.green-outline{border-color:rgba(60,190,122,0.45);color:var(--green);}
.btn.green-outline:hover{background:rgba(60,190,122,0.08);}
.btn.purple-outline{border-color:rgba(167,139,250,0.45);color:var(--purple);}
.btn.purple-outline:hover{background:rgba(167,139,250,0.08);}
.btn:disabled{opacity:.25;cursor:not-allowed;}
.icon-btn{background:transparent;border:none;cursor:pointer;padding:3px 5px;border-radius:4px;
  color:var(--text-muted);display:inline-flex;align-items:center;font-size:13px;
  transition:color .15s,background .15s;}
.icon-btn:hover{color:var(--text);background:rgba(255,255,255,0.06);}
.icon-btn.del:hover{color:var(--red);}

/* dataset list */
.ds-list{display:flex;flex-direction:column;gap:5px;}
.ds-card{background:var(--surface3);border:1px solid var(--border);border-radius:6px;
  cursor:pointer;transition:border-color .15s,background .15s;overflow:hidden;user-select:none;}
.ds-card:hover{border-color:var(--border-hi);}
.ds-card.active{border-color:rgba(167,139,250,0.55);background:rgba(167,139,250,0.04);}
.ds-top{display:flex;align-items:center;padding:8px 10px 5px;gap:8px;}
.ds-info{flex:1;min-width:0;}
.ds-name{font-family:'Barlow Condensed',sans-serif;font-size:12px;font-weight:600;
  letter-spacing:0.04em;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.ds-count{font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);}
.ds-badge{font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--purple);border:1px solid rgba(167,139,250,0.4);
  border-radius:3px;padding:1px 5px;white-space:nowrap;}
.ds-thumbs{display:flex;gap:2px;padding:0 10px 7px;}
.ds-thumb{width:30px;height:26px;object-fit:cover;border-radius:3px;flex-shrink:0;}
.ds-ph{width:30px;height:26px;background:var(--surface2);border-radius:3px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:9px;opacity:.3;}

/* dataset manage panel */
.ds-manage{display:none;border-top:1px solid var(--border);padding:8px 10px;background:var(--surface2);}
.ds-manage.open{display:block;}
.ds-manage-drop{border:1.5px dashed var(--border-hi);border-radius:5px;padding:10px;
  display:flex;flex-direction:column;align-items:center;gap:4px;cursor:pointer;
  transition:border-color .15s,background .15s;text-align:center;margin-bottom:8px;}
.ds-manage-drop:hover,.ds-manage-drop.over{border-color:rgba(167,139,250,0.6);background:rgba(167,139,250,0.05);}
.ds-manage-drop-lbl{font-family:'Barlow Condensed',sans-serif;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);}
.ds-img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(58px,1fr));gap:4px;max-height:180px;overflow-y:auto;}
.ds-img-grid::-webkit-scrollbar{width:2px;}.ds-img-grid::-webkit-scrollbar-thumb{background:var(--border-hi);}
.ds-img-item{position:relative;border-radius:4px;overflow:hidden;aspect-ratio:1;background:var(--surface);}
.ds-img-item img{width:100%;height:100%;object-fit:cover;display:block;}
.ds-img-del{position:absolute;top:2px;right:2px;background:rgba(0,0,0,0.7);border:none;
  border-radius:3px;color:#fff;font-size:10px;cursor:pointer;padding:1px 4px;
  opacity:0;transition:opacity .15s;line-height:1.4;}
.ds-img-item:hover .ds-img-del{opacity:1;}
.ds-img-item:hover .ds-img-del:hover{background:var(--red);}
.ds-manage-footer{display:flex;align-items:center;justify-content:space-between;margin-top:6px;}
.ds-manage-count{font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);}
.ulist-inline{display:flex;flex-direction:column;gap:2px;max-height:80px;overflow-y:auto;margin-bottom:6px;}
.ulist-inline::-webkit-scrollbar{width:2px;}.ulist-inline::-webkit-scrollbar-thumb{background:var(--border-hi);}

/* output cards */
.out-card{background:var(--surface3);border:1px solid var(--border);border-radius:6px;
  display:flex;align-items:center;gap:8px;padding:7px 9px;transition:border-color .15s;}
.out-card:hover{border-color:var(--border-hi);}
.out-thumb{width:36px;height:32px;object-fit:cover;border-radius:3px;flex-shrink:0;background:var(--surface2);}
.out-thumb-ph{width:36px;height:32px;background:var(--surface2);border-radius:3px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:11px;opacity:.3;}
.out-info{flex:1;min-width:0;}
.out-name{font-family:'Barlow Condensed',sans-serif;font-size:12px;font-weight:600;
  color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.out-meta{font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);}
.out-dl{background:var(--surface2);border:1px solid var(--border-hi);border-radius:5px;
  color:var(--green);font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:0.08em;
  padding:4px 10px;cursor:pointer;white-space:nowrap;transition:background .15s,border-color .15s;flex-shrink:0;}
.out-dl:hover{background:rgba(60,190,122,0.1);border-color:rgba(60,190,122,0.5);}
#outputsList::-webkit-scrollbar{width:3px;}
#outputsList::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:2px;}

/* new dataset panel */
.ndp{background:var(--surface3);border:1px solid var(--border);border-radius:6px;
  padding:10px;display:flex;flex-direction:column;gap:8px;}
.ndp.hidden{display:none;}
.drop-zone{border:1.5px dashed var(--border-hi);border-radius:6px;padding:16px 12px;
  display:flex;flex-direction:column;align-items:center;gap:5px;cursor:pointer;
  transition:border-color .15s,background .15s;text-align:center;}
.drop-zone:hover,.drop-zone.over{border-color:rgba(167,139,250,0.6);background:rgba(167,139,250,0.04);}
.dz-icon{font-size:20px;opacity:.45;}
.dz-lbl{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);}
.dz-sub{font-size:10px;color:var(--text-muted);opacity:.55;}
.ulist{display:flex;flex-direction:column;gap:3px;max-height:110px;overflow-y:auto;}
.ulist::-webkit-scrollbar{width:2px;}.ulist::-webkit-scrollbar-thumb{background:var(--border-hi);}
.uitem{display:flex;align-items:center;gap:6px;padding:3px 5px;background:var(--surface2);border-radius:4px;}
.uthumb{width:22px;height:18px;object-fit:cover;border-radius:2px;flex-shrink:0;background:var(--surface);}
.uname{font-size:10px;color:var(--text-dim);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.ust{font-size:10px;flex-shrink:0;color:var(--text-muted);}
.ust.ok{color:var(--green);}
.ust.err{color:var(--red);}
.ust.pend{animation:blink 1.2s infinite;color:var(--text-muted);}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* output display */
.out-display{font-size:11px;color:var(--text-muted);padding:5px 8px;background:var(--bg);
  border:1px solid var(--border);border-radius:5px;word-break:break-all;min-height:28px;line-height:1.4;}

/* sliders / checkboxes */
.sl{display:flex;flex-direction:column;gap:4px;}
.sl-h{display:flex;justify-content:space-between;align-items:center;}
input[type=range]{-webkit-appearance:none;width:100%;height:2px;background:var(--border-hi);border-radius:2px;outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--text);cursor:pointer;}
.sv{font-family:'Barlow Condensed',sans-serif;font-size:11px;color:var(--text-muted);}
.chks{display:flex;flex-wrap:wrap;gap:4px;}
.chk{display:flex;align-items:center;gap:5px;cursor:pointer;background:var(--bg);
  border:1px solid var(--border-hi);border-radius:5px;padding:4px 8px;user-select:none;
  transition:border-color .15s,background .15s;}
.chk input{display:none;}
.cb{width:11px;height:11px;border:1.5px solid var(--border-hi);border-radius:3px;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s;}
.chk input:checked~.cb{background:var(--text);border-color:var(--text);}
.chk input:checked~.cb::after{content:'';width:5px;height:3px;border-left:1.5px solid #080808;
  border-bottom:1.5px solid #080808;transform:rotate(-45deg) translate(1px,-1px);display:block;}
.chk:has(input:checked){border-color:rgba(255,255,255,0.28);background:rgba(255,255,255,0.04);}
.cl{font-family:'Outfit',sans-serif;font-size:11px;color:var(--text-muted);}
.chk:has(input:checked) .cl{color:var(--text);}

/* progress + run */
.prog{display:none;}.prog.show{display:flex;flex-direction:column;gap:4px;}
.prog-track{background:var(--border);border-radius:2px;height:2px;overflow:hidden;}
.prog-fill{height:100%;background:var(--text);transition:width .3s;width:0%;}
.prog-lbl{font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);}
.status-box{font-size:11px;color:var(--text-muted);padding:8px 10px;background:var(--bg);
  border:1px solid var(--border);border-radius:6px;min-height:50px;line-height:1.5;word-break:break-all;}
.run-btn{width:100%;padding:11px;background:var(--text);color:#080808;border:none;
  border-radius:var(--radius);font-family:'Barlow Condensed',sans-serif;font-size:13px;
  font-weight:600;letter-spacing:0.18em;text-transform:uppercase;cursor:pointer;
  transition:opacity .15s,transform .1s;margin-top:auto;}
.run-btn:hover{opacity:.85;transform:translateY(-1px);}
.run-btn:disabled{opacity:.25;cursor:not-allowed;transform:none;}
.run-btn.stop{background:var(--red);color:#fff;}

/* discord banner */
.discord-banner{display:flex;align-items:center;gap:10px;padding:10px 12px;
  background:rgba(88,101,242,0.08);border:1px solid rgba(88,101,242,0.3);border-radius:var(--radius);
  text-decoration:none;transition:background .15s,border-color .15s;flex-shrink:0;}
.discord-banner:hover{background:rgba(88,101,242,0.18);border-color:rgba(88,101,242,0.6);}
.discord-icon{width:22px;height:22px;color:#5865f2;flex-shrink:0;}
.discord-text{flex:1;min-width:0;}
.discord-title{font-family:'Barlow Condensed',sans-serif;font-size:12px;font-weight:600;
  letter-spacing:0.06em;color:var(--text);}
.discord-sub{font-size:10px;color:var(--text-muted);margin-top:2px;line-height:1.4;}
.discord-arrow{color:rgba(88,101,242,0.7);font-size:14px;flex-shrink:0;}
  background:rgba(88,101,242,0.1);border:1px solid rgba(88,101,242,0.35);
  border-radius:var(--radius);text-decoration:none;
  transition:background .15s,border-color .15s;cursor:pointer;}
.discord-banner:hover{background:rgba(88,101,242,0.18);border-color:rgba(88,101,242,0.6);}
.discord-icon{width:20px;height:20px;color:#5865f2;flex-shrink:0;}
.discord-text{flex:1;min-width:0;}
.discord-title{font-family:'Barlow Condensed',sans-serif;font-size:11px;font-weight:600;
  letter-spacing:0.1em;text-transform:uppercase;color:#7289da;}
.discord-sub{font-size:10px;color:var(--text-muted);margin-top:1px;}
.discord-arrow{color:rgba(88,101,242,0.6);font-size:14px;flex-shrink:0;}

/* center */
.main{display:flex;flex-direction:column;overflow:hidden;height:100%;}
.toolbar{display:flex;align-items:center;justify-content:space-between;
  padding:11px 16px;flex-shrink:0;border-bottom:1px solid var(--border);}
.tbl{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:var(--text-muted);}
.preview-strip{display:flex;gap:7px;padding:10px 16px;flex-shrink:0;overflow-x:auto;
  border-bottom:1px solid var(--border);min-height:76px;align-items:center;}
.preview-strip::-webkit-scrollbar{height:3px;}.preview-strip::-webkit-scrollbar-thumb{background:var(--border-hi);}
.preview-thumb{width:54px;height:54px;object-fit:cover;border-radius:5px;border:1px solid var(--border-hi);flex-shrink:0;}
.preview-empty{font-family:'Outfit',sans-serif;font-size:11px;color:var(--text-muted);text-align:center;width:100%;opacity:.7;}
.done-bar{margin:12px 16px 0;background:rgba(60,190,122,0.06);border:1px solid rgba(60,190,122,0.2);
  border-radius:var(--radius);padding:10px 14px;display:none;align-items:center;justify-content:space-between;gap:10px;flex-shrink:0;}
.done-bar.show{display:flex;}
.done-txt{font-family:'Barlow Condensed',sans-serif;font-size:11px;color:var(--green);}
.results-scroll{flex:1;overflow-y:auto;padding:12px 16px;}
.results-scroll::-webkit-scrollbar{width:4px;}.results-scroll::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:2px;}
.results-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;}
.empty-state{height:100%;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:8px;opacity:.16;padding:40px;}
.empty-state p{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--text-muted);text-align:center;}
.card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;animation:rise .2s ease;}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.card.processing{border-color:rgba(255,255,255,0.1);}
.card.done{border-color:rgba(60,190,122,0.18);}
.card.errored{border-color:rgba(224,60,60,0.18);}
.card-img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:var(--surface3);}
.card-img-ph{width:100%;aspect-ratio:4/3;background:var(--surface3);display:flex;align-items:center;justify-content:center;}
.card-body{padding:9px 11px;}
.card-name{font-family:'Barlow Condensed',sans-serif;font-size:9px;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;}
.card-st{display:flex;align-items:center;gap:5px;font-family:'Barlow Condensed',sans-serif;font-size:9px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;color:var(--text-muted);}
.sd{width:4px;height:4px;border-radius:50%;flex-shrink:0;}
.st-p .sd{background:var(--text-muted);animation:blink 1s infinite;}
.st-d .sd{background:var(--green);}
.st-e .sd{background:var(--red);}
.card-cap{font-size:11px;color:var(--text-muted);line-height:1.5;font-style:italic;min-height:28px;}
.card-cap.filled{color:var(--text-dim);font-style:normal;}

/* active dataset chip in sidebar */
.active-ds-chip{display:flex;align-items:center;gap:8px;padding:8px 10px;
  background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.3);border-radius:6px;}
.active-ds-chip.empty{background:var(--surface3);border-color:var(--border);}
.active-ds-name{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600;
  color:var(--text);flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.active-ds-count{font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--purple);white-space:nowrap;flex-shrink:0;}
.active-ds-none{font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);letter-spacing:0.08em;}

/* datasets modal */
.ds-modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);
  z-index:200;align-items:center;justify-content:center;padding:20px;}
.ds-modal-overlay.open{display:flex;}
.ds-modal{background:var(--surface);border:1px solid var(--border-hi);border-radius:12px;
  width:580px;max-width:100%;max-height:88vh;display:flex;flex-direction:column;overflow:hidden;}
.ds-modal-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid var(--border);flex-shrink:0;}
.ds-modal-title{font-family:'Barlow Condensed',sans-serif;font-size:13px;font-weight:600;
  letter-spacing:0.18em;text-transform:uppercase;}
.ds-modal-x{background:none;border:none;color:var(--text-muted);cursor:pointer;
  font-size:20px;line-height:1;padding:2px 6px;border-radius:4px;transition:color .15s,background .15s;}
.ds-modal-x:hover{color:var(--text);background:rgba(255,255,255,0.07);}
.ds-modal-body{overflow-y:auto;padding:16px 18px;display:flex;flex-direction:column;gap:10px;flex:1;}
.ds-modal-body::-webkit-scrollbar{width:3px;}
.ds-modal-body::-webkit-scrollbar-thumb{background:var(--border-hi);}

/* confirm dialog */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:300;align-items:center;justify-content:center;}
.overlay.open{display:flex;}
.dlg{background:var(--surface2);border:1px solid var(--border-hi);border-radius:10px;padding:20px;width:310px;max-width:95vw;}
.dlg-title{font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:0.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px;}
.dlg-msg{font-size:13px;line-height:1.5;margin-bottom:16px;}
.dlg-row{display:flex;gap:7px;justify-content:flex-end;}
</style>
</head>
<body>

<header class="topbar">
  <div class="brand-spacer"></div>
  <div class="topbar-logo">
    <img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCANQB9ADASIAAhEBAxEB/8QAHQABAAICAwEBAAAAAAAAAAAAAAECAwgFBgcECf/EAGQQAAEDAwEDBgcJCQsICAUEAwABAgMEBREGBxIhCBMxQVFhFBUiUnGBkRYyQoKSk5TR0hcYI1NVYnKVoSQzNTZFRlRWY3ODJUOEoqOxwdM0RHR1hbLC4QkmZGXiJzdHV/Bmw//EABgBAQEBAQEAAAAAAAAAAAAAAAABAgME/8QAIxEBAQEAAgICAwEBAQEAAAAAAAERAhIhURMxA0FhInEyQv/aAAwDAQACEQMRAD8A0yAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ4EACSAFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOoAAAAAAAAAAA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gSCMjIEgjIyBIIyMgSCBkCQRkZAkEZGQJBGRkCQRkZAkEZGQJBGRkCQRkZAkEZGQJBGRkCQRkgCxGSABORkgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAf/2Q==" alt="SKY" class="sky-logo">
  </div>
  <div class="pill" id="pill"><div class="pip"></div><span id="pillTxt">Model not loaded</span></div>
</header>

<!-- LEFT SIDEBAR -->
<aside class="sidebar-l">

  <!-- Datasets (compact — full management in modal) -->
  <div class="sec">
    <div class="sec-h">
      <div class="sec-h-l"><div class="sec-dot" style="background:var(--purple)"></div><span class="sec-title">Dataset</span></div>
      <button class="btn purple-outline" onclick="openDsModal()">&#9776; Manage</button>
    </div>
    <div class="sec-b">
      <div class="active-ds-chip empty" id="activeDsChip">
        <span class="active-ds-none">No dataset selected</span>
      </div>
    </div>
  </div>

  <!-- Captioned Outputs -->
  <div class="sec">
    <div class="sec-h">
      <div class="sec-h-l"><div class="sec-dot" style="background:var(--green)"></div><span class="sec-title">Captioned Outputs</span></div>
      <button class="btn" onclick="loadOutputs()" id="refreshOutputsBtn" title="Refresh">&#8635;</button>
    </div>
    <div class="sec-b" style="gap:6px;">
      <div id="outputsList" style="display:flex;flex-direction:column;gap:5px;max-height:220px;overflow-y:auto;padding-right:2px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);text-align:center;padding:10px;opacity:.6;">No captioned datasets yet</div>
      </div>
    </div>
  </div>

  <!-- Trigger word -->
  <div class="sec">
    <div class="sec-h">
      <div class="sec-h-l"><div class="sec-dot" style="background:var(--purple)"></div><span class="sec-title">LoRA / Trigger</span></div>
    </div>
    <div class="sec-b">
      <div class="field">
        <div class="lbl" style="color:var(--purple)">Trigger Word <span style="color:var(--red)">*</span></div>
        <input type="text" id="lora_trigger" placeholder="e.g. abc123 (required)" oninput="onTriggerChange()" style="border-color:rgba(167,139,250,0.3)">
        <div id="trigger_hint" style="font-size:10px;color:var(--red);display:none;">&#9888; Required to run</div>
      </div>
      <div class="field">
        <div class="lbl">Append Tag</div>
        <input type="text" id="append_tag" placeholder="e.g. 3mm4" oninput="saveConfig()">
      </div>
      <div class="field">
        <div class="lbl">Output Folder</div>
        <div class="out-display" id="outDisplay"><em style="opacity:.35">/workspace/captioned/{trigger}</em></div>
      </div>
    </div>
  </div>

  <!-- Caption style -->
  <div class="sec">
    <div class="sec-h"><div class="sec-h-l"><div class="sec-dot" style="background:var(--amber)"></div><span class="sec-title">Caption Style</span></div></div>
    <div class="sec-b">
      <div class="row2">
        <div class="field"><div class="lbl">Type</div>
          <select id="caption_type" onchange="saveConfig()">
            <option value="Training Prompt" selected>Training Prompt</option>
            <option value="Descriptive">Descriptive</option>
            <option value="Descriptive (Casual)">Descriptive (Casual)</option>
            <option value="Straightforward">Straightforward</option>
            <option value="Stable Diffusion">Stable Diffusion</option>
            <option value="MidJourney">MidJourney</option>
          </select>
        </div>
        <div class="field"><div class="lbl">Length</div>
          <select id="caption_length" onchange="saveConfig()">
            <option value="short">Short</option>
            <option value="medium" selected>Medium</option>
            <option value="long">Long</option>
          </select>
        </div>
      </div>
      <div class="field">
        <div class="lbl">Word Count</div>
        <select id="word_count" onchange="saveConfig()">
          <option value="none">No limit</option>
          <option value="15 words">15 words</option>
          <option value="20 words">20 words</option>
          <option value="30 words" selected>30 words</option>
          <option value="40 words">40 words</option>
          <option value="50 words">50 words</option>
          <option value="75 words">75 words</option>
          <option value="100 words">100 words</option>
        </select>
      </div>
      <div class="sl">
        <div class="sl-h"><div class="lbl">Max Tokens</div><span class="sv" id="sv_tok">400</span></div>
        <input type="range" id="max_new_tokens" min="100" max="800" step="25" value="400"
          oninput="document.getElementById('sv_tok').textContent=this.value;saveConfig()">
      </div>
      <div class="row2">
        <div class="sl">
          <div class="sl-h"><div class="lbl">Temp</div><span class="sv" id="sv_temp">1.00</span></div>
          <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="1.0"
            oninput="document.getElementById('sv_temp').textContent=parseFloat(this.value).toFixed(2);saveConfig()">
        </div>
        <div class="sl">
          <div class="sl-h"><div class="lbl">Top P</div><span class="sv" id="sv_topp">0.90</span></div>
          <input type="range" id="top_p" min="0.1" max="1.0" step="0.05" value="0.9"
            oninput="document.getElementById('sv_topp').textContent=parseFloat(this.value).toFixed(2);saveConfig()">
        </div>
      </div>
    </div>
  </div>

</aside>

<!-- CENTER -->
<main class="main">
  <div class="toolbar">
    <span class="tbl" id="toolbar_count">—</span>
    <button class="btn primary" id="dl_btn" onclick="downloadAll()" disabled>&#8595; Download ZIP</button>
  </div>
  <div class="preview-strip" id="preview_strip">
    <div class="preview-empty">Select or create a dataset to begin</div>
  </div>
  <div class="done-bar" id="done_bar">
    <span class="done-txt" id="done_txt"></span>
    <button class="btn primary" onclick="downloadAll()">&#8595; Download .zip</button>
  </div>
  <div class="results-scroll">
    <div class="empty-state" id="empty_state"><p>Results appear here<br>as images are captioned</p></div>
    <div class="results-grid" id="results_grid" style="display:none"></div>
  </div>
</main>

<!-- RIGHT SIDEBAR -->
<aside class="sidebar-r">
  <div class="sec">
    <div class="sec-h"><div class="sec-h-l"><div class="sec-dot" style="background:var(--green)"></div><span class="sec-title">Include</span></div></div>
    <div class="sec-b">
      <div class="chks">
        <label class="chk"><input type="checkbox" id="lighting" onchange="saveConfig()"><div class="cb"></div><span class="cl">Lighting</span></label>
        <label class="chk"><input type="checkbox" id="camera_angle" onchange="saveConfig()"><div class="cb"></div><span class="cl">Camera Angle</span></label>
        <label class="chk"><input type="checkbox" id="vantage_height" onchange="saveConfig()"><div class="cb"></div><span class="cl">Cam Height</span></label>
        <label class="chk"><input type="checkbox" id="shot_type" onchange="saveConfig()"><div class="cb"></div><span class="cl">Shot Type</span></label>
        <label class="chk"><input type="checkbox" id="light_sources" onchange="saveConfig()"><div class="cb"></div><span class="cl">Light Sources</span></label>
        <label class="chk"><input type="checkbox" id="char_age" onchange="saveConfig()"><div class="cb"></div><span class="cl">Age</span></label>
        <label class="chk"><input type="checkbox" id="nsfw" checked onchange="saveConfig()"><div class="cb"></div><span class="cl">NSFW</span></label>
        <label class="chk"><input type="checkbox" id="no_euphemisms" checked onchange="saveConfig()"><div class="cb"></div><span class="cl">No Euphemisms</span></label>
      </div>
    </div>
  </div>
  <div class="sec">
    <div class="sec-h"><div class="sec-h-l"><div class="sec-dot" style="background:#60a5fa"></div><span class="sec-title">Attribute Mentions</span></div></div>
    <div class="sec-b">
      <div class="chks">
        <label class="chk"><input type="checkbox" id="mention_age" onchange="saveConfig()"><div class="cb"></div><span class="cl">Age</span></label>
        <label class="chk"><input type="checkbox" id="mention_hair_color" onchange="saveConfig()"><div class="cb"></div><span class="cl">Hair Color</span></label>
        <label class="chk"><input type="checkbox" id="mention_hair_length" onchange="saveConfig()"><div class="cb"></div><span class="cl">Hair Length</span></label>
        <label class="chk"><input type="checkbox" id="mention_hair_style" onchange="saveConfig()"><div class="cb"></div><span class="cl">Hair Style</span></label>
        <label class="chk"><input type="checkbox" id="mention_eye_color" onchange="saveConfig()"><div class="cb"></div><span class="cl">Eye Color</span></label>
        <label class="chk"><input type="checkbox" id="mention_body_type" onchange="saveConfig()"><div class="cb"></div><span class="cl">Body Type</span></label>
      </div>
      <div class="field" style="margin-top:4px;">
        <div class="lbl">Age Override</div>
        <input type="text" id="age_override" placeholder="e.g. young adult, adult" oninput="saveConfig()">
      </div>
      <div class="field">
        <div class="lbl">Do Not Mention</div>
        <input type="text" id="do_not_mention" placeholder="comma separated" oninput="saveConfig()">
      </div>
      <div class="chks" style="margin-top:2px;">
        <label class="chk"><input type="checkbox" id="exclude_ethnicity" checked onchange="saveConfig()"><div class="cb"></div><span class="cl">Exclude Ethnicity</span></label>
      </div>
    </div>
  </div>
  <div class="prog" id="prog">
    <div class="prog-track"><div class="prog-fill" id="prog_fill"></div></div>
    <div class="prog-lbl" id="prog_lbl">0 / 0</div>
  </div>
  <div class="status-box" id="status">Ready.</div>

  <a href="https://discord.com/invite/loras" target="_blank" class="discord-banner">
    <svg class="discord-icon" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M20.317 4.492c-1.53-.69-3.17-1.2-4.885-1.49a.075.075 0 0 0-.079.036c-.21.369-.444.85-.608 1.23a18.566 18.566 0 0 0-5.487 0 12.36 12.36 0 0 0-.617-1.23A.077.077 0 0 0 8.562 3c-1.714.29-3.354.8-4.885 1.491a.07.07 0 0 0-.032.027C.533 9.093-.32 13.555.099 17.961a.08.08 0 0 0 .031.055 20.03 20.03 0 0 0 5.993 2.98.078.078 0 0 0 .084-.026c.462-.62.874-1.275 1.226-1.963.021-.04.001-.088-.041-.104a13.201 13.201 0 0 1-1.872-.878.075.075 0 0 1-.008-.125c.126-.093.252-.19.372-.287a.075.075 0 0 1 .078-.01c3.927 1.764 8.18 1.764 12.061 0a.075.075 0 0 1 .079.009c.12.098.245.195.372.288a.075.075 0 0 1-.006.125c-.598.344-1.22.635-1.873.877a.075.075 0 0 0-.041.105c.36.687.772 1.341 1.225 1.962a.077.077 0 0 0 .084.028 19.963 19.963 0 0 0 6.002-2.981.076.076 0 0 0 .032-.054c.5-5.094-.838-9.52-3.549-13.442a.06.06 0 0 0-.031-.028zM8.02 15.278c-1.182 0-2.157-1.069-2.157-2.38 0-1.312.956-2.38 2.157-2.38 1.21 0 2.176 1.077 2.157 2.38 0 1.312-.956 2.38-2.157 2.38zm7.975 0c-1.183 0-2.157-1.069-2.157-2.38 0-1.312.955-2.38 2.157-2.38 1.21 0 2.176 1.077 2.157 2.38 0 1.312-.946 2.38-2.157 2.38z"/>
    </svg>
    <div class="discord-text">
      <div class="discord-title">Want more LoRAs &amp; workflows?</div>
      <div class="discord-sub">Join our Discord to browse, download &amp; purchase exclusive LoRAs and ComfyUI workflows.</div>
    </div>
    <div class="discord-arrow">&#8599;</div>
  </a>

  <button class="run-btn" id="runBtn" onclick="run()">&#9654;&nbsp; Run Captioning</button>
</aside>

<!-- Datasets Modal -->
<div class="ds-modal-overlay" id="dsModalOverlay" onclick="if(event.target===this)closeDsModal()">
  <div class="ds-modal">
    <div class="ds-modal-hdr">
      <span class="ds-modal-title">&#9776; Datasets</span>
      <div style="display:flex;gap:8px;align-items:center;">
        <button class="btn purple-outline" id="modalNewDsBtn" onclick="toggleNewDs()">＋ New Dataset</button>
        <button class="ds-modal-x" onclick="closeDsModal()">&#10005;</button>
      </div>
    </div>
    <div class="ds-modal-body">

      <!-- New dataset panel -->
      <div class="ndp hidden" id="ndp">
        <div class="field">
          <div class="lbl">Dataset Name</div>
          <input type="text" id="ndpName" placeholder="e.g. mymodel_v1" onkeydown="if(event.key==='Enter')createDs()">
        </div>
        <div class="drop-zone" id="dropZone"
          onclick="document.getElementById('fileInput').click()"
          ondragover="dzOver(event)" ondragleave="dzLeave(event)" ondrop="dzDrop(event)">
          <div class="dz-icon">&#128247;</div>
          <div class="dz-lbl">Drop images here</div>
          <div class="dz-sub">or click to browse &nbsp;·&nbsp; JPG PNG WEBP BMP TIFF</div>
        </div>
        <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff" style="display:none" onchange="onFilePick(event)">
        <div class="ulist" id="ulist"></div>
        <div style="display:flex;gap:6px;margin-top:2px;">
          <button class="btn primary" onclick="createDs()" style="flex:1">Create &amp; Upload</button>
          <button class="btn" onclick="cancelDs()">Cancel</button>
        </div>
      </div>

      <!-- Dataset list -->
      <div class="ds-list" id="dsList">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);text-align:center;padding:16px;opacity:.6;">Loading...</div>
      </div>

    </div>
  </div>
</div>

<!-- Delete confirm -->
<div class="overlay" id="delOverlay">
  <div class="dlg">
    <div class="dlg-title">Delete Dataset</div>
    <div class="dlg-msg" id="delMsg"></div>
    <div class="dlg-row">
      <button class="btn" onclick="closeDelDlg()">Cancel</button>
      <button class="btn" style="border-color:rgba(224,60,60,0.5);color:var(--red);" onclick="doDelete()">Delete</button>
    </div>
  </div>
</div>

<script>
var running=false, abortCtrl=null, saveTimer=null;
var activeDs=null;          // {name, path, count}
var lastOutputFolder='/workspace/captioned';
var pendingDeleteName=null;
var pendingFiles=[];

// ── Boot ──────────────────────────────────────────────────────────────────────
fetch('/api/status').then(r=>r.json()).then(d=>{if(d.model_loaded)setReady();});
loadConfig();
loadDatasets();
loadOutputs();

// ── Config ────────────────────────────────────────────────────────────────────
function loadConfig(){
  fetch('/api/config').then(r=>r.json()).then(cfg=>{
    function setV(id,v){var e=document.getElementById(id);if(!e)return;
      if(e.type==='checkbox')e.checked=!!v;else if(e.type==='range')e.value=v;else e.value=v||'';}
    ['lora_trigger','append_tag','caption_type','caption_length','word_count',
     'max_new_tokens','temperature','top_p','lighting','camera_angle','vantage_height',
     'shot_type','light_sources','char_age','nsfw','no_euphemisms','mention_age',
     'mention_hair_color','mention_hair_length','mention_hair_style','mention_eye_color',
     'mention_body_type','age_override','exclude_ethnicity','do_not_mention']
    .forEach(k=>setV(k,cfg[k]));
    document.getElementById('sv_tok').textContent=cfg.max_new_tokens||400;
    document.getElementById('sv_temp').textContent=parseFloat(cfg.temperature||1).toFixed(2);
    document.getElementById('sv_topp').textContent=parseFloat(cfg.top_p||0.9).toFixed(2);
    updateOutDisplay(cfg.output_folder||'');
    lastOutputFolder=cfg.output_folder||'/workspace/captioned';
    if(cfg.active_dataset) setTimeout(()=>selectDsByName(cfg.active_dataset),500);
  }).catch(()=>{});
}

function getCfg(){
  function gv(id){var e=document.getElementById(id);if(!e)return '';
    if(e.type==='checkbox')return e.checked;if(e.type==='range')return parseFloat(e.value);return e.value.trim();}
  return{
    active_dataset:activeDs?activeDs.name:'',
    dataset_folder:activeDs?activeDs.path:'',
    lora_trigger:gv('lora_trigger'), append_tag:gv('append_tag'),
    caption_type:gv('caption_type'), caption_length:gv('caption_length'),
    word_count:gv('word_count'), max_new_tokens:parseInt(gv('max_new_tokens')),
    temperature:parseFloat(gv('temperature')), top_p:parseFloat(gv('top_p')),
    lighting:gv('lighting'), camera_angle:gv('camera_angle'),
    vantage_height:gv('vantage_height'), shot_type:gv('shot_type'),
    light_sources:gv('light_sources'), char_age:gv('char_age'),
    nsfw:gv('nsfw'), no_euphemisms:gv('no_euphemisms'),
    mention_age:gv('mention_age'), mention_hair_color:gv('mention_hair_color'),
    mention_hair_length:gv('mention_hair_length'), mention_hair_style:gv('mention_hair_style'),
    mention_eye_color:gv('mention_eye_color'), mention_body_type:gv('mention_body_type'),
    age_override:gv('age_override'), exclude_ethnicity:gv('exclude_ethnicity'),
    do_not_mention:gv('do_not_mention'),
  };
}

function saveConfig(){
  clearTimeout(saveTimer);
  saveTimer=setTimeout(()=>{
    fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getCfg())})
      .then(r=>r.json()).then(d=>{if(d.output_folder){updateOutDisplay(d.output_folder);lastOutputFolder=d.output_folder;}})
      .catch(()=>{});
  },300);
}

function updateOutDisplay(folder){
  var el=document.getElementById('outDisplay');
  var trigger=document.getElementById('lora_trigger').value.trim();
  el.innerHTML=trigger&&folder
    ? '<span style="color:var(--green)">'+esc(folder)+'</span>'
    : '<em style="opacity:.35">/workspace/captioned/{trigger}</em>';
}

function onTriggerChange(){
  var t=document.getElementById('lora_trigger').value.trim();
  if(t){document.getElementById('trigger_hint').style.display='none';
    document.getElementById('lora_trigger').style.borderColor='rgba(167,139,250,0.3)';}
  else updateOutDisplay('');
  saveConfig();
}

// ── Datasets ──────────────────────────────────────────────────────────────────
function loadDatasets(){
  fetch('/api/datasets').then(r=>r.json()).then(d=>renderDatasets(d.datasets||[])).catch(()=>{});
}

function loadOutputs(){
  var btn=document.getElementById('refreshOutputsBtn');
  if(btn)btn.style.opacity='0.4';
  fetch('/api/outputs').then(r=>r.json()).then(d=>{
    renderOutputs(d.outputs||[]);
    if(btn)btn.style.opacity='1';
  }).catch(()=>{if(btn)btn.style.opacity='1';});
}

function renderOutputs(list){
  var el=document.getElementById('outputsList');
  if(!list.length){
    el.innerHTML='<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:10px;color:var(--text-muted);text-align:center;padding:10px;opacity:.6;">No captioned datasets yet</div>';
    return;
  }
  el.innerHTML='';
  list.forEach(out=>{
    var card=document.createElement('div');
    card.className='out-card';
    var thumbHtml=out.thumb
      ? `<img class="out-thumb" src="${out.thumb}">`
      : `<div class="out-thumb-ph">📄</div>`;
    card.innerHTML=`
      ${thumbHtml}
      <div class="out-info">
        <div class="out-name">${esc(out.name)}</div>
        <div class="out-meta">${out.images} img &nbsp;·&nbsp; ${out.texts} captions</div>
      </div>
      <button class="out-dl" onclick="downloadOutput('${esc(out.path)}','${esc(out.name)}')">&#8595; ZIP</button>`;
    el.appendChild(card);
  });
}

function downloadOutput(path, name){
  setStatus('Preparing download for "'+name+'"...');
  window.location.href='/api/download?folder='+encodeURIComponent(path);
  setTimeout(()=>setStatus('Ready.'), 2000);
}

function renderDatasets(list){
  var el=document.getElementById('dsList');
  if(!list.length){
    el.innerHTML='<div style="font-family:\'Barlow Condensed\',sans-serif;font-size:10px;color:var(--text-muted);text-align:center;padding:12px;opacity:.6;">No datasets — create one above</div>';
    return;
  }
  // Remember which cards were open
  var openPanels={};
  document.querySelectorAll('.ds-manage.open').forEach(p=>{openPanels[p.dataset.ds]=true;});
  el.innerHTML='';
  list.forEach(ds=>{
    var active=activeDs&&activeDs.name===ds.name;
    var card=document.createElement('div');
    card.className='ds-card'+(active?' active':'');
    card.id='dsc-'+ds.name;
    var thumbs=ds.thumbs.length
      ? ds.thumbs.map(t=>`<img class="ds-thumb" src="${t}">`).join('')
      : '<div class="ds-ph">🖼</div>';
    card.innerHTML=`
      <div class="ds-top">
        <div class="ds-info">
          <div class="ds-name">${esc(ds.name)}</div>
          <div class="ds-count" id="dscnt-${esc(ds.name)}">${ds.count} image${ds.count===1?'':'s'}</div>
        </div>
        ${active?'<span class="ds-badge">Active</span>':''}
        <div style="display:flex;gap:2px;align-items:center;flex-shrink:0;">
          <button class="icon-btn" title="Add / manage images" onclick="toggleManage('${esc(ds.name)}',event)" id="mgbtn-${esc(ds.name)}">&#9881;</button>
          <button class="icon-btn del" title="Delete dataset" onclick="askDelete('${esc(ds.name)}',event)">🗑</button>
        </div>
      </div>
      <div class="ds-thumbs">${thumbs}</div>
      <div class="ds-manage${openPanels[ds.name]?' open':''}" id="mgpanel-${esc(ds.name)}" data-ds="${esc(ds.name)}">
        <div class="ds-manage-drop" id="mgdrop-${esc(ds.name)}"
          onclick="document.getElementById('mgfile-${esc(ds.name)}').click()"
          ondragover="mgDzOver(event,'${esc(ds.name)}')" ondragleave="mgDzLeave('${esc(ds.name)}')" ondrop="mgDzDrop(event,'${esc(ds.name)}')">
          <div class="dz-icon" style="font-size:16px;">&#43;</div>
          <div class="ds-manage-drop-lbl">Add more images</div>
        </div>
        <input type="file" id="mgfile-${esc(ds.name)}" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff" style="display:none"
          onchange="mgFilePick(event,'${esc(ds.name)}')">
        <div class="ulist-inline" id="mgulist-${esc(ds.name)}"></div>
        <div class="ds-img-grid" id="mgimgs-${esc(ds.name)}">
          <div style="grid-column:1/-1;font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);padding:4px 0;">Loading images...</div>
        </div>
        <div class="ds-manage-footer">
          <span class="ds-manage-count" id="mgcount-${esc(ds.name)}">${ds.count} images</span>
        </div>
      </div>`;
    card.addEventListener('click',e=>{
      if(e.target.closest('.icon-btn')||e.target.closest('.ds-manage'))return;
      selectDs(ds);
    });
    el.appendChild(card);
    if(openPanels[ds.name]) loadManageImages(ds.name);
  });
}

function openDsModal(){
  document.getElementById('dsModalOverlay').className='ds-modal-overlay open';
  loadDatasets();
}
function closeDsModal(){
  document.getElementById('dsModalOverlay').className='ds-modal-overlay';
  cancelDs();
}

function updateActiveDsChip(){
  var chip=document.getElementById('activeDsChip');
  if(!chip)return;
  if(activeDs){
    chip.className='active-ds-chip';
    chip.innerHTML=`<span class="active-ds-name">${esc(activeDs.name)}</span><span class="active-ds-count">${activeDs.count} img</span>`;
  } else {
    chip.className='active-ds-chip empty';
    chip.innerHTML='<span class="active-ds-none">No dataset selected</span>';
  }
}

function selectDs(ds){
  activeDs=ds;
  document.querySelectorAll('.ds-card').forEach(c=>c.classList.remove('active'));
  var c=document.getElementById('dsc-'+ds.name);
  if(c)c.classList.add('active');
  // Always sync trigger word + output to selected dataset
  var tEl=document.getElementById('lora_trigger');
  tEl.value=ds.name;
  onTriggerChange();
  scanPreview(ds.path);
  saveConfig();
  updateActiveDsChip();
  setStatus('Dataset "'+ds.name+'" selected ('+ds.count+' images)');
  closeDsModal();
}

function selectDsByName(name){
  fetch('/api/datasets').then(r=>r.json()).then(d=>{
    var ds=(d.datasets||[]).find(x=>x.name===name);
    if(ds){
      activeDs=ds;
      updateActiveDsChip();
      renderDatasets(d.datasets);
      scanPreview(ds.path);
    }
  });
}

function scanPreview(path){
  if(!path)return;
  fetch('/api/scan?path='+encodeURIComponent(path)).then(r=>r.json()).then(d=>{
    var strip=document.getElementById('preview_strip');
    document.getElementById('toolbar_count').textContent=d.valid&&d.count?d.count+' images in dataset':'No images';
    if(d.valid&&d.count){
      strip.innerHTML='';
      d.previews.forEach(p=>{var img=document.createElement('img');img.className='preview-thumb';img.src=p.thumb;img.title=p.name;strip.appendChild(img);});
      if(d.count>d.previews.length){var m=document.createElement('div');m.style.cssText='font-family:Barlow Condensed,sans-serif;font-size:10px;color:var(--text-muted);white-space:nowrap;align-self:center;';m.textContent='+'+(d.count-d.previews.length)+' more';strip.appendChild(m);}
    } else strip.innerHTML='<div class="preview-empty">Dataset is empty — upload images above</div>';
  });
}

// ── New Dataset Panel ─────────────────────────────────────────────────────────
function toggleNewDs(){
  var p=document.getElementById('ndp');
  p.classList.contains('hidden')?p.classList.remove('hidden'):cancelDs();
  if(!p.classList.contains('hidden'))setTimeout(()=>document.getElementById('ndpName').focus(),50);
}
function cancelDs(){
  var p=document.getElementById('ndp');if(p)p.classList.add('hidden');
  var n=document.getElementById('ndpName');if(n)n.value='';
  var u=document.getElementById('ulist');if(u)u.innerHTML='';
  pendingFiles=[];
}

function dzOver(e){e.preventDefault();document.getElementById('dropZone').classList.add('over');}
function dzLeave(){document.getElementById('dropZone').classList.remove('over');}
function dzDrop(e){
  e.preventDefault();dzLeave();
  addFiles(Array.from(e.dataTransfer.files).filter(f=>/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name)));
}
function onFilePick(e){addFiles(Array.from(e.target.files));e.target.value='';}

function safeId(name){return name.replace(/[^a-zA-Z0-9_-]/g,'_');}

function addFiles(files){
  files.forEach(f=>{if(!pendingFiles.find(p=>p.name===f.name&&p.size===f.size))pendingFiles.push(f);});
  renderPending();
}
function renderPending(){
  var ul=document.getElementById('ulist');ul.innerHTML='';
  pendingFiles.forEach(f=>{
    var sid=safeId(f.name);
    var item=document.createElement('div');item.className='uitem';item.id='ui-'+sid;
    item.innerHTML=`<img class="uthumb" id="uth-${sid}"><span class="uname">${esc(f.name)}</span><span class="ust pend" id="ust-${sid}">queued</span>`;
    if(f.type.startsWith('image/')){var r=new FileReader();r.onload=ev=>{var el=document.getElementById('uth-'+sid);if(el)el.src=ev.target.result;};r.readAsDataURL(f);}
    ul.appendChild(item);
  });
}

function createDs(){
  var name=document.getElementById('ndpName').value.trim();
  if(!name){document.getElementById('ndpName').focus();setStatus('⚠ Enter a dataset name first.');return;}
  if(!pendingFiles.length){setStatus('⚠ Drop some images in first, or create empty dataset.');
    // Allow creating empty dataset anyway
  }
  setStatus('Creating dataset "'+name+'"...');
  fetch('/api/dataset/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})
    .then(r=>{
      if(!r.ok)return r.json().then(e=>{throw new Error(e.error||'Server error '+r.status);});
      return r.json();
    })
    .then(d=>{
      if(!d.ok)throw new Error(d.error||'Failed to create dataset');
      if(!pendingFiles.length){
        setStatus('Dataset "'+d.name+'" created.');
        cancelDs();
        fetch('/api/datasets').then(r=>r.json()).then(res=>{
          renderDatasets(res.datasets||[]);
          var ds=(res.datasets||[]).find(x=>x.name===d.name);
          if(ds){
            activeDs=ds;
            document.getElementById('lora_trigger').value=ds.name;
            onTriggerChange();
            updateActiveDsChip();
            scanPreview(ds.path);
            saveConfig();
          }
          closeDsModal();
        });
        return;
      }
      uploadFiles(d.name, pendingFiles, ()=>{
        fetch('/api/datasets').then(r=>r.json()).then(res=>{
          renderDatasets(res.datasets||[]);
          var ds=(res.datasets||[]).find(x=>x.name===d.name);
          if(ds){
            activeDs=ds;
            document.getElementById('lora_trigger').value=ds.name;
            onTriggerChange();
            updateActiveDsChip();
            scanPreview(ds.path);
            saveConfig();
          }
          cancelDs();
          closeDsModal();
        });
      });
    })
    .catch(err=>{
      setStatus('⚠ Error: '+err.message);
      console.error('createDs error:',err);
    });
}

function uploadFiles(dsName, files, onDone){
  var ulist=document.getElementById('ulist');
  if(ulist){
    ulist.innerHTML='<div style="font-family:Barlow Condensed,sans-serif;font-size:10px;color:var(--text-muted);margin-bottom:4px;">Uploading <span id=\"uploadDone\">0</span> / '+files.length+' files</div><div style="background:var(--border);border-radius:2px;height:3px;overflow:hidden;"><div id=\"uploadBar\" style=\"height:100%;background:var(--purple);width:0%;transition:width .2s;\"></div></div>';
  }
  setStatus('Uploading '+files.length+' file'+(files.length===1?'':'s')+' to "'+dsName+'"...');
  var BATCH=20, done=0, errs=0;
  function updateProg(){
    var pct=Math.round(done/files.length*100);
    var bar=document.getElementById('uploadBar');
    var lbl=document.getElementById('uploadDone');
    if(bar)bar.style.width=pct+'%';
    if(lbl)lbl.textContent=done;
  }
  function batch(i){
    if(i>=files.length){
      if(errs>0)setStatus('Upload done: '+done+' saved, '+errs+' failed.');
      else setStatus('✓ Uploaded '+done+' image'+(done===1?'':'s')+' to "'+dsName+'"');
      if(onDone)onDone();return;
    }
    var b=files.slice(i,i+BATCH), fd=new FormData();
    fd.append('dataset',dsName);
    b.forEach(f=>fd.append('files',f));
    fetch('/api/upload',{method:'POST',body:fd})
      .then(r=>{
        if(!r.ok)throw new Error('Server error '+r.status);
        return r.json();
      })
      .then(res=>{
        done+=(res.saved||[]).length;
        errs+=(res.errors||[]).length;
        updateProg();
        batch(i+BATCH);
      })
      .catch(err=>{
        errs+=b.length; done+=b.length;
        updateProg();
        setStatus('⚠ Upload error: '+err.message);
        batch(i+BATCH);
      });
  }
  batch(0);
}

// ── Manage Panel (add/remove images) ─────────────────────────────────────────
function toggleManage(dsName, e){
  e.stopPropagation();
  var panel=document.getElementById('mgpanel-'+dsName);
  if(!panel)return;
  var isOpen=panel.classList.contains('open');
  // Close all other panels first
  document.querySelectorAll('.ds-manage.open').forEach(p=>p.classList.remove('open'));
  if(!isOpen){
    panel.classList.add('open');
    loadManageImages(dsName);
  }
}

function loadManageImages(dsName){
  var grid=document.getElementById('mgimgs-'+dsName);
  if(!grid)return;
  grid.innerHTML='<div style="grid-column:1/-1;font-family:\'Barlow Condensed\',sans-serif;font-size:10px;color:var(--text-muted);padding:4px 0;">Loading...</div>';
  fetch('/api/dataset/images?name='+encodeURIComponent(dsName))
    .then(r=>r.json()).then(d=>{
      renderManageImages(dsName, d.images||[]);
    }).catch(()=>{grid.innerHTML='<div style="grid-column:1/-1;font-size:10px;color:var(--red);">Error loading images</div>';});
}

function renderManageImages(dsName, images){
  var grid=document.getElementById('mgimgs-'+dsName);
  var countEl=document.getElementById('mgcount-'+dsName);
  if(!grid)return;
  if(!images.length){
    grid.innerHTML='<div style="grid-column:1/-1;font-family:\'Barlow Condensed\',sans-serif;font-size:10px;color:var(--text-muted);padding:4px 0;opacity:.6;">No images yet</div>';
    if(countEl)countEl.textContent='0 images';
    return;
  }
  grid.innerHTML='';
  images.forEach(img=>{
    var item=document.createElement('div');
    item.className='ds-img-item';
    item.id='imgitem-'+dsName+'-'+img.name;
    item.innerHTML=`<img src="${img.thumb}" title="${esc(img.name)}">`
      +`<button class="ds-img-del" onclick="deleteImage('${esc(dsName)}','${esc(img.name)}',event)" title="Remove">✕</button>`;
    grid.appendChild(item);
  });
  if(countEl)countEl.textContent=images.length+' image'+(images.length===1?'':'s');
  // Also update card count label
  var cntEl=document.getElementById('dscnt-'+dsName);
  if(cntEl)cntEl.textContent=images.length+' image'+(images.length===1?'':'s');
}

function deleteImage(dsName, imgName, e){
  e.stopPropagation();
  var item=document.getElementById('imgitem-'+dsName+'-'+imgName);
  if(item)item.style.opacity='0.3';
  fetch('/api/dataset/image/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({dataset:dsName,image:imgName})})
    .then(r=>r.json()).then(d=>{
      if(d.ok){
        if(item)item.remove();
        var countEl=document.getElementById('mgcount-'+dsName);
        if(countEl)countEl.textContent=d.remaining+' image'+(d.remaining===1?'':'s');
        var cntEl=document.getElementById('dscnt-'+dsName);
        if(cntEl)cntEl.textContent=d.remaining+' image'+(d.remaining===1?'':'s');
        // Refresh preview if this is the active dataset
        if(activeDs&&activeDs.name===dsName){
          activeDs.count=d.remaining;
          scanPreview(activeDs.path);
        }
      } else {
        if(item)item.style.opacity='1';
        setStatus('Error: '+d.error);
      }
    }).catch(()=>{if(item)item.style.opacity='1';});
}

// Drag & drop for manage panel
function mgDzOver(e,dsName){e.preventDefault();e.stopPropagation();var d=document.getElementById('mgdrop-'+dsName);if(d)d.classList.add('over');}
function mgDzLeave(dsName){var d=document.getElementById('mgdrop-'+dsName);if(d)d.classList.remove('over');}
function mgDzDrop(e,dsName){
  e.preventDefault();e.stopPropagation();mgDzLeave(dsName);
  var files=Array.from(e.dataTransfer.files).filter(f=>/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name));
  if(files.length)mgUpload(dsName,files);
}
function mgFilePick(e,dsName){
  var files=Array.from(e.target.files);
  e.target.value='';
  if(files.length)mgUpload(dsName,files);
}

function mgUpload(dsName, files){
  var ulist=document.getElementById('mgulist-'+safeId(dsName));
  if(!ulist) ulist=document.getElementById('mgulist-'+dsName);
  if(ulist){
    ulist.innerHTML='';
    files.forEach(f=>{
      var sid=safeId(dsName)+'-'+safeId(f.name);
      var item=document.createElement('div');item.className='uitem';
      item.innerHTML=`<span class="uname">${esc(f.name)}</span><span class="ust pend" id="mgust-${sid}">uploading</span>`;
      ulist.appendChild(item);
    });
  }
  var BATCH=10, done=0, errs=0;
  function batch(i){
    if(i>=files.length){
      if(errs>0)setStatus('Added '+done+' image'+(done===1?'':'s')+' to "'+dsName+'" ('+errs+' failed)');
      else setStatus('✓ Added '+done+' image'+(done===1?'':'s')+' to "'+dsName+'"');
      loadManageImages(dsName);
      if(activeDs&&activeDs.name===dsName)scanPreview(activeDs.path);
      setTimeout(()=>{if(ulist)ulist.innerHTML='';},2000);
      return;
    }
    var b=files.slice(i,i+BATCH), fd=new FormData();
    fd.append('dataset',dsName);
    b.forEach(f=>fd.append('files',f));
    fetch('/api/upload',{method:'POST',body:fd})
      .then(r=>{
        if(!r.ok)throw new Error('Server error '+r.status);
        return r.json();
      })
      .then(res=>{
        (res.saved||[]).forEach(s=>{
          done++;
          var origFile=b.find(f=>f.name===s.name||safeId(f.name)===safeId(s.name));
          var sid=safeId(dsName)+'-'+(origFile?safeId(origFile.name):safeId(s.name));
          var el=document.getElementById('mgust-'+sid);
          if(el){el.textContent='✓';el.className='ust ok';}
        });
        errs+=(res.errors||[]).length;
        batch(i+BATCH);
      })
      .catch(err=>{
        errs+=b.length;
        b.forEach(f=>{var el=document.getElementById('mgust-'+safeId(dsName)+'-'+safeId(f.name));if(el){el.textContent='✗';el.className='ust err';}});
        setStatus('⚠ Upload error: '+err.message);
        console.error('mgUpload error:',err);
        batch(i+BATCH);
      });
  }
  batch(0);
}

// ── Delete dataset ────────────────────────────────────────────────────────────
function askDelete(name,e){
  e.stopPropagation();pendingDeleteName=name;
  document.getElementById('delMsg').textContent='Delete "'+name+'" and all its images? This cannot be undone.';
  document.getElementById('delOverlay').className='overlay open';
}
function closeDelDlg(){document.getElementById('delOverlay').className='overlay';pendingDeleteName=null;}
function doDelete(){
  if(!pendingDeleteName){closeDelDlg();return;}
  fetch('/api/dataset/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:pendingDeleteName})})
    .then(r=>r.json()).then(()=>{
      closeDelDlg();
      if(activeDs&&activeDs.name===pendingDeleteName){
        activeDs=null;
        updateActiveDsChip();
        document.getElementById('preview_strip').innerHTML='<div class="preview-empty">Select or create a dataset to begin</div>';
        document.getElementById('toolbar_count').textContent='—';
      }
      loadDatasets();setStatus('Dataset deleted.');
    });
}

// ── Run ───────────────────────────────────────────────────────────────────────
function run(){
  if(running){stop();return;}
  var cfg=getCfg();
  if(!cfg.lora_trigger){
    document.getElementById('trigger_hint').style.display='block';
    document.getElementById('lora_trigger').style.borderColor='var(--red)';
    document.getElementById('lora_trigger').focus();
    setStatus('⚠ Enter a Trigger Word to run.');return;
  }
  if(!cfg.dataset_folder){setStatus('⚠ Select a dataset first.');return;}
  running=true;lastOutputFolder='/workspace/captioned/'+cfg.lora_trigger;
  document.getElementById('runBtn').textContent='◼  Stop';
  document.getElementById('runBtn').className='run-btn stop';
  document.getElementById('prog').className='prog show';
  document.getElementById('prog_fill').style.width='0%';
  document.getElementById('done_bar').className='done-bar';
  document.getElementById('dl_btn').disabled=true;
  document.getElementById('results_grid').innerHTML='';
  document.getElementById('results_grid').style.display='none';
  document.getElementById('empty_state').style.display='flex';
  abortCtrl=new AbortController();var buf='';
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg),signal:abortCtrl.signal})
    .then(r=>{
      if(!r.ok&&r.status===400)return r.json().then(e=>{setStatus('⚠ '+e.error);finalize();throw new Error('blocked');});
      var rdr=r.body.getReader(),dec=new TextDecoder();
      function read(){rdr.read().then(chunk=>{
        if(chunk.done){finalize();return;}
        buf+=dec.decode(chunk.value,{stream:true});
        var lines=buf.split('\n');buf=lines.pop();
        lines.forEach(l=>{if(l.startsWith('data: ')){try{onEv(JSON.parse(l.slice(6)));}catch(e){}}});
        read();
      }).catch(()=>finalize());}
      read();
    }).catch(e=>{if(e.name!=='AbortError'&&e.message!=='blocked')setStatus('Error: '+e.message);finalize();});
}
function stop(){if(abortCtrl)abortCtrl.abort();finalize();}
function finalize(){running=false;document.getElementById('runBtn').textContent='▶\u00a0 Run Captioning';document.getElementById('runBtn').className='run-btn';}
function onEv(d){
  if(d.type==='log')setStatus(d.msg);
  else if(d.type==='model_ready'){setReady();setStatus('Model loaded — starting...');}
  else if(d.type==='start')setStatus('Captioning '+d.total+' images → '+d.output+'...');
  else if(d.type==='processing'){setProg(d.i-1,d.total);addCard(d.out_name,d.name,d.thumb,'processing');}
  else if(d.type==='done'){setProg(d.i,d.total);updCard(d.out_name,'done',d.caption);}
  else if(d.type==='err_one')updCard(d.out_name,'errored','⚠ '+d.error);
  else if(d.type==='error'){setStatus('Error: '+d.msg);finalize();}
  else if(d.type==='complete'){
    setProg(d.total,d.total);
    var m=Math.floor(d.elapsed/60),s=d.elapsed%60;
    setStatus('Done! '+d.total+' images captioned in '+m+'m '+s+'s');
    document.getElementById('dl_btn').disabled=false;
    document.getElementById('done_txt').textContent='✓ '+d.total+' captions → '+d.output;
    document.getElementById('done_bar').className='done-bar show';
    lastOutputFolder=d.output;finalize();
    loadOutputs(); // refresh outputs panel
  }
}
function setProg(n,t){var p=t>0?Math.round(n/t*100):0;document.getElementById('prog_fill').style.width=p+'%';document.getElementById('prog_lbl').textContent=n+' / '+t+' ('+p+'%)';}
function addCard(id,name,thumb,st){
  document.getElementById('empty_state').style.display='none';document.getElementById('results_grid').style.display='grid';
  var g=document.getElementById('results_grid'),d=document.createElement('div');
  d.className='card '+st;d.id='card-'+id;
  d.innerHTML=(thumb?`<img class="card-img" src="${thumb}">`:'<div class="card-img-ph"><span style="font-size:20px;opacity:.3">🖼</span></div>')
    +`<div class="card-body"><div class="card-name">${esc(name)}</div>`
    +`<div class="card-st st-p" id="st-${id}"><div class="sd"></div><span>Generating</span></div>`
    +`<div class="card-cap" id="cap-${id}"></div></div>`;
  g.appendChild(d);d.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function updCard(id,st,cap){
  var c=document.getElementById('card-'+id);if(c)c.className='card '+st;
  var s=document.getElementById('st-'+id);
  if(s){if(st==='done'){s.className='card-st st-d';s.innerHTML='<div class="sd"></div><span>Done</span>';}
    if(st==='errored'){s.className='card-st st-e';s.innerHTML='<div class="sd"></div><span>Error</span>';}}
  var cp=document.getElementById('cap-'+id);if(cp){cp.className='card-cap filled';cp.textContent=cap;}
}
function downloadAll(){window.location.href='/api/download?folder='+encodeURIComponent(lastOutputFolder);}
function setReady(){var p=document.getElementById('pill');p.className='pill ready';document.getElementById('pillTxt').textContent='Model ready';}
function setStatus(m){document.getElementById('status').textContent=m;}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n" + "="*44)
    print("  SkyCaption")
    print("  Open port 5000 in RunPod Connect")
    print("="*44 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
