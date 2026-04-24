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
  <div class="brand"><div class="brand-dot"></div>SkyCaption</div>
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

    <!-- VIEW: LIST -->
    <div class="dsv" id="dsv-list">
      <div class="ds-modal-hdr">
        <span class="ds-modal-title">&#9776; Datasets</span>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn purple-outline" onclick="dsvShow('create')">&#65291; New</button>
          <button class="ds-modal-x" onclick="closeDsModal()">&#10005;</button>
        </div>
      </div>
      <div class="ds-modal-body">
        <div class="ds-list" id="dsList">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);text-align:center;padding:16px;opacity:.6;">Loading...</div>
        </div>
      </div>
    </div>

    <!-- VIEW: CREATE / UPLOAD -->
    <div class="dsv hidden" id="dsv-create">
      <div class="ds-modal-hdr">
        <div style="display:flex;align-items:center;gap:8px;">
          <button class="ds-back-btn" id="createBackBtn" onclick="cancelCreate()">&#8592;</button>
          <span class="ds-modal-title">New Dataset</span>
        </div>
        <button class="ds-modal-x" onclick="closeDsModal()">&#10005;</button>
      </div>
      <div class="ds-modal-body">
        <div class="field">
          <div class="lbl">Dataset Name</div>
          <input type="text" id="ndpName" placeholder="e.g. mymodel_v1"
            onkeydown="if(event.key==='Enter')createDs()">
        </div>
        <div class="drop-zone" id="dropZone"
          onclick="document.getElementById('fileInput').click()"
          ondragover="dzOver(event)" ondragleave="dzLeave(event)" ondrop="dzDrop(event)">
          <div class="dz-icon">&#128247;</div>
          <div class="dz-lbl" id="dzLbl">Drop images here</div>
          <div class="dz-sub">or click to browse &nbsp;&#183;&nbsp; JPG PNG WEBP BMP TIFF</div>
        </div>
        <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif"
          style="display:none" onchange="onFilePick(event)">
        <div class="upload-summary hidden" id="uploadSummary">
          <span class="us-count" id="usSummaryCount">0 files</span>
          <button class="us-clear" onclick="clearPendingFiles()">&#10005; Clear</button>
        </div>
        <div class="upload-progress hidden" id="uploadProgress">
          <div class="up-track"><div class="up-fill" id="upFill"></div></div>
          <div class="up-stats">
            <span><span id="upDone">0</span>&thinsp;/&thinsp;<span id="upTotal">0</span> uploaded</span>
            <span id="upErrs" style="color:var(--red);display:none;"></span>
          </div>
        </div>
        <div class="create-done hidden" id="createDone">
          <span class="cd-icon">&#10003;</span>
          <span class="cd-msg" id="cdMsg"></span>
        </div>
        <div style="display:flex;gap:6px;" id="createFooter">
          <button class="btn primary" id="createBtn" onclick="createDs()" style="flex:1">Create &amp; Upload</button>
          <button class="btn" id="cancelCreateBtn" onclick="cancelCreate()">Cancel</button>
        </div>
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
          <div class="ds-manage-drop-lbl">Drop or click to add images</div>
        </div>
        <input type="file" id="mgfile-${esc(ds.name)}" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif" style="display:none"
          onchange="mgFilePick(event,'${esc(ds.name)}')">
        <div class="mg-up-wrap hidden" id="mgprog-${safeId(ds.name)}">
          <div class="mg-up-track"><div class="mg-up-fill" id="mgfill-${safeId(ds.name)}"></div></div>
          <div class="mg-up-lbl" id="mglbl-${safeId(ds.name)}"></div>
        </div>
        <div class="ds-img-grid" id="mgimgs-${esc(ds.name)}">
          <div style="grid-column:1/-1;font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);padding:4px 0;">Loading images...</div>
        </div>
        <div class="ds-manage-footer">
          <span class="ds-manage-count" id="mgcount-${esc(ds.name)}">${ds.count} images</span>
        </div>
        </div>`"""
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
  <div class="brand"><div class="brand-dot"></div>SkyCaption</div>
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

    <!-- VIEW: LIST -->
    <div class="dsv" id="dsv-list">
      <div class="ds-modal-hdr">
        <span class="ds-modal-title">&#9776; Datasets</span>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn purple-outline" onclick="dsvShow('create')">&#65291; New</button>
          <button class="ds-modal-x" onclick="closeDsModal()">&#10005;</button>
        </div>
      </div>
      <div class="ds-modal-body">
        <div class="ds-list" id="dsList">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:10px;color:var(--text-muted);text-align:center;padding:16px;opacity:.6;">Loading...</div>
        </div>
      </div>
    </div>

    <!-- VIEW: CREATE / UPLOAD -->
    <div class="dsv hidden" id="dsv-create">
      <div class="ds-modal-hdr">
        <div style="display:flex;align-items:center;gap:8px;">
          <button class="ds-back-btn" id="createBackBtn" onclick="cancelCreate()">&#8592;</button>
          <span class="ds-modal-title">New Dataset</span>
        </div>
        <button class="ds-modal-x" onclick="closeDsModal()">&#10005;</button>
      </div>
      <div class="ds-modal-body">
        <div class="field">
          <div class="lbl">Dataset Name</div>
          <input type="text" id="ndpName" placeholder="e.g. mymodel_v1"
            onkeydown="if(event.key==='Enter')createDs()">
        </div>
        <div class="drop-zone" id="dropZone"
          onclick="document.getElementById('fileInput').click()"
          ondragover="dzOver(event)" ondragleave="dzLeave(event)" ondrop="dzDrop(event)">
          <div class="dz-icon">&#128247;</div>
          <div class="dz-lbl" id="dzLbl">Drop images here</div>
          <div class="dz-sub">or click to browse &nbsp;&#183;&nbsp; JPG PNG WEBP BMP TIFF</div>
        </div>
        <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.tif"
          style="display:none" onchange="onFilePick(event)">
        <div class="upload-summary hidden" id="uploadSummary">
          <span class="us-count" id="usSummaryCount">0 files</span>
          <button class="us-clear" onclick="clearPendingFiles()">&#10005; Clear</button>
        </div>
        <div class="upload-progress hidden" id="uploadProgress">
          <div class="up-track"><div class="up-fill" id="upFill"></div></div>
          <div class="up-stats">
            <span><span id="upDone">0</span>&thinsp;/&thinsp;<span id="upTotal">0</span> uploaded</span>
            <span id="upErrs" style="color:var(--red);display:none;"></span>
          </div>
        </div>
        <div class="create-done hidden" id="createDone">
          <span class="cd-icon">&#10003;</span>
          <span class="cd-msg" id="cdMsg"></span>
        </div>
        <div style="display:flex;gap:6px;" id="createFooter">
          <button class="btn primary" id="createBtn" onclick="createDs()" style="flex:1">Create &amp; Upload</button>
          <button class="btn" id="cancelCreateBtn" onclick="cancelCreate()">Cancel</button>
        </div>
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

// ── Dataset Modal ─────────────────────────────────────────────────────────────
function openDsModal(){
  dsvShow('list');
  document.getElementById('dsModalOverlay').className='ds-modal-overlay open';
  loadDatasets();
}
function closeDsModal(){
  document.getElementById('dsModalOverlay').className='ds-modal-overlay';
  pendingFiles=[];
}

function dsvShow(view){
  document.getElementById('dsv-list').classList.toggle('hidden', view!=='list');
  document.getElementById('dsv-create').classList.toggle('hidden', view!=='create');
  if(view==='create'){
    pendingFiles=[];
    // Reset all create view states
    document.getElementById('ndpName').value='';
    document.getElementById('ndpName').disabled=false;
    document.getElementById('dropZone').style.display='';
    document.getElementById('uploadSummary').classList.add('hidden');
    document.getElementById('uploadProgress').classList.add('hidden');
    document.getElementById('createDone').classList.add('hidden');
    document.getElementById('createFooter').style.display='';
    document.getElementById('createBtn').disabled=false;
    document.getElementById('createBtn').textContent='Create & Upload';
    document.getElementById('createBackBtn').style.display='';
    setTimeout(()=>document.getElementById('ndpName').focus(),60);
  }
}

function cancelCreate(){
  pendingFiles=[];
  dsvShow('list');
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
    if(ds){ activeDs=ds; updateActiveDsChip(); renderDatasets(d.datasets); scanPreview(ds.path); }
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

// ── File Handling ─────────────────────────────────────────────────────────────
function dzOver(e){e.preventDefault();document.getElementById('dropZone').classList.add('over');}
function dzLeave(e){document.getElementById('dropZone').classList.remove('over');}
function dzDrop(e){
  e.preventDefault();dzLeave(e);
  addFiles(Array.from(e.dataTransfer.files).filter(f=>/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name)));
}
function onFilePick(e){addFiles(Array.from(e.target.files));e.target.value='';}

function safeId(name){return name.replace(/[^a-zA-Z0-9_-]/g,'_');}

function addFiles(files){
  files.forEach(f=>{if(!pendingFiles.find(p=>p.name===f.name&&p.size===f.size))pendingFiles.push(f);});
  updateSummary();
}

function clearPendingFiles(){
  pendingFiles=[];
  updateSummary();
}

function updateSummary(){
  var sum=document.getElementById('uploadSummary');
  var cnt=document.getElementById('usSummaryCount');
  if(!sum)return;
  if(pendingFiles.length){
    cnt.textContent=pendingFiles.length+' file'+(pendingFiles.length===1?'':'s')+' ready';
    sum.classList.remove('hidden');
  } else {
    sum.classList.add('hidden');
  }
}

// ── Upload Engine (parallel) ──────────────────────────────────────────────────
async function uploadEngine(dsName, files, onProgress, onDone){
  var CONCURRENCY=4, BATCH=5;
  var batchIdx=0, done=0, errors=0;
  var total=files.length;
  var batches=[];
  for(var i=0;i<files.length;i+=BATCH) batches.push(files.slice(i,i+BATCH));

  async function worker(){
    while(true){
      var myIdx=batchIdx++;
      if(myIdx>=batches.length) break;
      var batch=batches[myIdx];
      var fd=new FormData();
      fd.append('dataset',dsName);
      batch.forEach(f=>fd.append('files',f));
      try{
        var r=await fetch('/api/upload',{method:'POST',body:fd});
        if(!r.ok) throw new Error('HTTP '+r.status);
        var d=await r.json();
        done+=(d.saved||[]).length;
        errors+=(d.errors||[]).length;
      }catch(e){
        errors+=batch.length;
        done+=batch.length;
      }
      onProgress(done,total,errors);
    }
  }

  onProgress(0,total,0);
  var n=Math.min(CONCURRENCY,Math.max(1,batches.length));
  await Promise.all(Array.from({length:n},()=>worker()));
  onDone(done,errors);
}

// ── Create Dataset ────────────────────────────────────────────────────────────
function refreshAfterUpload(dsName){
  fetch('/api/datasets').then(r=>r.json()).then(res=>{
    renderDatasets(res.datasets||[]);
    var ds=(res.datasets||[]).find(x=>x.name===dsName);
    if(ds){
      activeDs=ds;
      document.getElementById('lora_trigger').value=ds.name;
      onTriggerChange();
      updateActiveDsChip();
      scanPreview(ds.path);
      saveConfig();
    }
  });
}

function createDs(){
  var name=document.getElementById('ndpName').value.trim();
  if(!name){document.getElementById('ndpName').focus();setStatus('⚠ Enter a dataset name first.');return;}

  // Lock the form
  document.getElementById('createBtn').disabled=true;
  document.getElementById('createBtn').textContent='Creating...';
  document.getElementById('ndpName').disabled=true;
  document.getElementById('createBackBtn').style.display='none';
  document.getElementById('cancelCreateBtn').disabled=true;

  fetch('/api/dataset/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})
    .then(r=>{if(!r.ok)return r.json().then(e=>{throw new Error(e.error||'Server error '+r.status);});return r.json();})
    .then(d=>{
      if(!d.ok) throw new Error(d.error||'Failed to create dataset');

      if(!pendingFiles.length){
        // Empty dataset created
        setStatus('Dataset "'+d.name+'" created.');
        refreshAfterUpload(d.name);
        dsvShow('list');
        return;
      }

      // Switch UI to upload mode
      document.getElementById('dropZone').style.display='none';
      document.getElementById('uploadSummary').classList.add('hidden');
      document.getElementById('cancelCreateBtn').disabled=true;
      var upEl=document.getElementById('uploadProgress');
      upEl.classList.remove('hidden');
      document.getElementById('upTotal').textContent=pendingFiles.length;
      document.getElementById('upDone').textContent='0';
      document.getElementById('upFill').style.width='0%';
      document.getElementById('upErrs').style.display='none';
      document.getElementById('createBtn').textContent='Uploading...';
      setStatus('Uploading '+pendingFiles.length+' file'+(pendingFiles.length===1?'':'s')+' to "'+d.name+'"...');

      var filesToUpload=pendingFiles.slice();
      uploadEngine(d.name, filesToUpload,
        function(done,total,errors){
          var pct=total?Math.round(done/total*100):0;
          document.getElementById('upFill').style.width=pct+'%';
          document.getElementById('upDone').textContent=done;
          if(errors>0){var e=document.getElementById('upErrs');e.style.display='';e.textContent=errors+' failed';}
        },
        function(done,errors){
          pendingFiles=[];
          var msg=done+' image'+(done===1?'':'s')+' uploaded'+(errors?' ('+errors+' failed)':'');
          setStatus('✓ '+msg+' to "'+d.name+'"');
          document.getElementById('uploadProgress').classList.add('hidden');
          document.getElementById('createDone').classList.remove('hidden');
          document.getElementById('cdMsg').textContent=msg;
          document.getElementById('createFooter').style.display='none';
          refreshAfterUpload(d.name);
          setTimeout(()=>{ closeDsModal(); },1800);
        }
      );
    })
    .catch(err=>{
      setStatus('⚠ Error: '+err.message);
      // Restore form
      document.getElementById('createBtn').disabled=false;
      document.getElementById('createBtn').textContent='Create & Upload';
      document.getElementById('ndpName').disabled=false;
      document.getElementById('createBackBtn').style.display='';
      document.getElementById('cancelCreateBtn').disabled=false;
    });
}

// ── Manage Panel (add/remove images) ─────────────────────────────────────────
function toggleManage(dsName, e){
  e.stopPropagation();
  var panel=document.getElementById('mgpanel-'+dsName);
  if(!panel)return;
  var isOpen=panel.classList.contains('open');
  document.querySelectorAll('.ds-manage.open').forEach(p=>p.classList.remove('open'));
  if(!isOpen){ panel.classList.add('open'); loadManageImages(dsName); }
}

function loadManageImages(dsName){
  var grid=document.getElementById('mgimgs-'+dsName);
  if(!grid)return;
  grid.innerHTML='<div style="grid-column:1/-1;font-family:\'Barlow Condensed\',sans-serif;font-size:10px;color:var(--text-muted);padding:4px 0;">Loading...</div>';
  fetch('/api/dataset/images?name='+encodeURIComponent(dsName))
    .then(r=>r.json()).then(d=>{ renderManageImages(dsName, d.images||[]); })
    .catch(()=>{grid.innerHTML='<div style="grid-column:1/-1;font-size:10px;color:var(--red);">Error loading images</div>';});
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
        if(activeDs&&activeDs.name===dsName){ activeDs.count=d.remaining; scanPreview(activeDs.path); }
      } else {
        if(item)item.style.opacity='1';
        setStatus('Error: '+d.error);
      }
    }).catch(()=>{if(item)item.style.opacity='1';});
}

// Manage panel drag & drop
function mgDzOver(e,dsName){e.preventDefault();e.stopPropagation();var d=document.getElementById('mgdrop-'+dsName);if(d)d.classList.add('over');}
function mgDzLeave(dsName){var d=document.getElementById('mgdrop-'+dsName);if(d)d.classList.remove('over');}
function mgDzDrop(e,dsName){
  e.preventDefault();e.stopPropagation();mgDzLeave(dsName);
  var files=Array.from(e.dataTransfer.files).filter(f=>/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name));
  if(files.length) mgUpload(dsName,files);
}
function mgFilePick(e,dsName){
  var files=Array.from(e.target.files);
  e.target.value='';
  if(files.length) mgUpload(dsName,files);
}

function mgUpload(dsName, files){
  if(!files.length)return;
  var sid=safeId(dsName);
  var progEl=document.getElementById('mgprog-'+sid);
  var fillEl=document.getElementById('mgfill-'+sid);
  var lblEl=document.getElementById('mglbl-'+sid);
  var dropEl=document.getElementById('mgdrop-'+dsName);

  if(progEl) progEl.classList.remove('hidden');
  if(dropEl) dropEl.style.opacity='0.4';
  if(lblEl) lblEl.textContent='0 / '+files.length;

  uploadEngine(dsName, files,
    function(done,total,errors){
      var pct=total?Math.round(done/total*100):0;
      if(fillEl) fillEl.style.width=pct+'%';
      if(lblEl) lblEl.textContent=done+' / '+total+(errors?' (⚠ '+errors+' err)':'');
    },
    function(done,errors){
      setStatus('✓ Added '+done+' image'+(done===1?'':'s')+' to "'+dsName+'"'+(errors?' ('+errors+' failed)':''));
      loadManageImages(dsName);
      if(activeDs&&activeDs.name===dsName) scanPreview(activeDs.path);
      setTimeout(()=>{
        if(progEl) progEl.classList.add('hidden');
        if(fillEl) fillEl.style.width='0%';
        if(dropEl) dropEl.style.opacity='';
      },1500);
    }
  );
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
