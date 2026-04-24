# SkyCaption — Docker Setup

## How to build (using GitHub Actions — no local Docker needed)

### Step 1 — Create a GitHub repo
1. Go to github.com → New repository → name it `skycaption` → Public → Create
2. Upload ALL files from this zip into the repo (drag and drop them in the GitHub UI)
   Make sure the `.github/workflows/` folder is included

### Step 2 — Add Docker Hub secrets to GitHub
1. In your GitHub repo go to **Settings → Secrets and variables → Actions → New repository secret**
2. Add these two secrets:
   - Name: `DOCKER_USERNAME` — Value: `ueso3vji4`
   - Name: `DOCKER_PASSWORD` — Value: your Docker Hub password

### Step 3 — Trigger the build
1. Go to your repo → **Actions** tab
2. Click **Build & Push to Docker Hub** → **Run workflow** → **Run workflow**
3. It will take ~10-15 minutes to build (downloading the PyTorch base image)
4. When it goes green ✅ your image is live at `ueso3vji4/skycaption:latest`

### Step 4 — Deploy on RunPod
1. RunPod → **Deploy** → **Custom Image**
2. Container image: `ueso3vji4/skycaption:latest`
3. GPU: 16GB+ VRAM (RTX 3090, 4090, A4000 etc.)
4. Volume mount: `/workspace` (30GB+)
5. Expose port: `5000`
6. Deploy — first boot downloads the model (~7GB) to the volume, restarts are instant

## File structure
```
skycaption/
├── .github/
│   └── workflows/
│       └── docker-build.yml   ← auto-builds on every push
├── app.py                     ← SkyCaption app
├── Dockerfile
├── entrypoint.sh
├── requirements.txt
└── .dockerignore
```
