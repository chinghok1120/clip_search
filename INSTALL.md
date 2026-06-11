# Installation Guide

This guide covers installation for both development (Linux server with GPU) and deployment (Jetson Orin Nano).

## Directory Structure Philosophy

**Git repo** (source) and **deployment folder** (runtime) are kept separate:

- **Git repo**: Full repository with all code, docs, web UI, etc.
- **Deployment folder**: Clean `pn/` contents only, no git history

---

## Option 1: Development on Linux Server

For developing the web UI, testing models, or contributing to the project.

### Setup

```bash
# Clone the repository
cd ~/projects  # or your preferred location
git clone https://github.com/chinghok1120/clip_search.git
cd clip_search

# Install dependencies for web UI
cd web/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the web interface
./start.sh
# Access at http://localhost:8000
```

### Deploy to Jetson from Linux server

```bash
# Preview what will be deployed
./deploy.sh

# Actually deploy to Jetson
./deploy.sh --go

# Deploy to different Jetson
PN_HOST=user@192.168.1.100 ./deploy.sh --go
```

---

## Option 2: Direct Installation on Jetson

For running the PN (Processing Node) directly on the Jetson, or for studying the code on the Jetson itself.

### Directory Structure

When you clone and deploy locally, the script automatically creates separate folders:

```
~/clip_search/          ← Full git clone (source code)
~/clip_search_deploy/   ← Deployed pn/ folder (runtime) - auto-created
```

**Note:** If git repo and deploy target conflict, deploy script automatically adjusts the target to `*_deploy` to keep them separate.

### Setup

```bash
# 1. Clone the repository (use default name)
cd ~
git clone https://github.com/chinghok1120/clip_search.git
cd clip_search

# 2. Deploy locally (creates ~/clip_search_deploy/ automatically)
./deploy.sh --local --go

# 3. Run setup scripts in deployment folder
cd ~/clip_search_deploy
./setup/setup_model.sh    # Build TensorRT engine (~5 min, needs 16GB RAM)
./setup/setup_db.sh       # Install FAISS

# 4. Provide thumbnail images (example with CrowdHuman dataset)
# Put JPEGs at: ~/datasets/crowdhuman/train/images_960
# Or run: ./tools/resize_images.py on your raw dataset

# 5. Run the demo
./run_demo.sh
# Access at http://<jetson-ip>:8000
```

### Update after git pull

```bash
# Update source repo
cd ~/clip_search
git pull

# Re-deploy to runtime folder
./deploy.sh --local --go

# Artifacts (venv, *.engine, embeddings) are preserved automatically
# Only code is updated
```

---

## Option 3: Remote-Only (No Git on Jetson)

If you don't want git on the Jetson at all, deploy from your dev machine:

```bash
# On Linux server
cd ~/projects/clip_search
git pull
./deploy.sh --go

# Then SSH to Jetson and run setup
ssh superrx@210.17.139.83
cd ~/clip_search
./setup/setup_model.sh
./setup/setup_db.sh
./run_demo.sh
```

---

## Deploy Script Reference

### Remote Deploy (from dev machine to Jetson)
```bash
./deploy.sh              # Dry-run preview
./deploy.sh --go         # Actually deploy
PN_HOST=user@host ./deploy.sh --go  # Custom target
```

### Local Deploy (on Jetson, repo → deployment folder)
```bash
./deploy.sh --local      # Dry-run preview
./deploy.sh --local --go # Actually deploy
PN_HOST=localhost ./deploy.sh --go  # Alternative syntax
```

### What Gets Deployed

**Included** (from `pn/`):
- All Python scripts
- Setup scripts (`setup/`)
- Tools (`tools/`, `bench/`)
- README and documentation

**Excluded** (artifact layer, see `pn/.deployignore`):
- `venv/` (virtual environment)
- `*.pth`, `*.engine` (model files)
- `embeddings/` (FAISS indices)
- Benchmark data and caches

Artifacts are built ON the Jetson by the setup scripts, never copied.

---

## Prerequisites

### For Linux Server (Web UI)
- Python 3.8+
- CUDA-capable GPU (NVIDIA RTX recommended)
- ~10GB disk space

### For Jetson (PN Deployment)
- **CRITICAL**: Jetson Orin Nano 16GB with **JetPack flashed**
  - JetPack includes CUDA PyTorch + TensorRT (DO NOT pip install torch)
- `git` (for Option 2 only)
- ≥5GB free disk space in `$HOME`
- ≥16GB RAM (or close GUI to free memory during setup)

---

## Troubleshooting

### Jetson setup_model.sh fails with "Killed"
**Cause**: Out of memory during TensorRT conversion

**Solutions**:
1. Close GUI and heavy applications
2. Increase swap space
3. Run in headless mode (no desktop environment)

```bash
# Check memory
free -h

# Kill heavy processes
killall gnome-shell  # If running GUI
```

### rsync: connection refused
**Cause**: SSH not configured or wrong hostname

**Solution**: Set up SSH key or check PN_HOST
```bash
ssh-copy-id superrx@210.17.139.83
```

### Custom deployment location
By default, local deploy creates `~/clip_search_deploy/`. To customize:
```bash
# Deploy to custom location
PN_DIR=my_custom_folder ./deploy.sh --local --go
cd ~/my_custom_folder
./setup/setup_model.sh
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Clone repo | `git clone https://github.com/chinghok1120/clip_search.git` |
| Deploy remote | `./deploy.sh --go` |
| Deploy local | `./deploy.sh --local --go` |
| Setup Jetson model | `cd ~/clip_search && ./setup/setup_model.sh` |
| Setup Jetson DB | `cd ~/clip_search && ./setup/setup_db.sh` |
| Run demo | `cd ~/clip_search && ./run_demo.sh` |
| Run web UI | `cd web && ./start.sh` |

---

Last Updated: 2026-06-11
