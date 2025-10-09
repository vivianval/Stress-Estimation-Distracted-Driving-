# #!/bin/bash

# echo "Please add the following to your ~/.bashrc"
# echo "export PATH=$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin"
# echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib:/usr/local/lib"


# #To get CUDA
# #wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
# #sudo sh cuda_11.7.0_515.43.04_linux.run

# sudo apt install pip cmake python3-venv build-essential libpthread-stubs0-dev 

# python3 -m venv virtual

# source virtual/bin/activate #<--- To antistoixo conda activate



# pip install --upgrade pip

# #pip install -r requirements311.txt

# pip install -r requirementsGithub.txt
# print('yes iam ')

# pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2

# pip install pandas==1.4.2 numpy==1.21.0 scikit-video==1.1.11
#!/usr/bin/env bash
set -euo pipefail

# ---- Env exports (append once) ----
if ! grep -q 'CUDA/bin' "$HOME/.bashrc"; then
  {
    echo ''
    echo '# >>> emoca env >>>'
    echo 'export PATH="$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH:/usr/local/cuda/bin"'
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib:/usr/local/lib"'
    echo '# <<< emoca env <<<'
  } >> "$HOME/.bashrc"
  echo "Added PATH and LD_LIBRARY_PATH to ~/.bashrc (open a new shell to pick them up)."
else
  echo "~/.bashrc already contains CUDA PATH entries."
fi

# ---- Base packages ----
sudo apt-get update
sudo apt-get install -y build-essential libpthread-stubs0-dev cmake python3-venv python3-pip

# ---- Virtualenv (prefer python3.9 if available) ----
PY=python3
if command -v python3.9 >/dev/null 2>&1; then PY=python3.9; fi

if [ ! -d virtual ]; then
  "$PY" -m venv virtual
fi

# # ---- Activate venv ----
# # shellcheck disable=SC1091
# source virtual/bin/activate

# python -m pip install --upgrade pip setuptools wheel

# # ---- Install requirements (skip chumpy which is problematic) ----
# TMP_REQ=$(mktemp)
# grep -v -E '^\s*chumpy(~=|==)|^#\s*chumpy' requirementsGithub.txt > "$TMP_REQ"

# grep -v -E '^( *#)?\s*(mediapipe|onnxruntime(-gpu)?|flatbuffers)([<>=].*)?$' requirementsGithub.txt > /tmp/req_core.txt
# pip install -r "$TMP_REQ"

# # 2) Add mediapipe with a newer flatbuffers, and (optionally) onnxruntime cpu
# pip install "flatbuffers>=2,<3" "mediapipe==0.10.20" "onnxruntime==1.17.*"


# ---- Activate venv ----
source virtual/bin/activate
python -m pip install --upgrade pip setuptools wheel

# ---- Prepare a constraints file with safe pins ----
CONSTR=$(mktemp)
cat > "$CONSTR" <<'EOF'
# Force modern flatbuffers compatible with mediapipe 0.10.x
flatbuffers>=2,<3
# Use CPU ONNX Runtime to avoid old GPU wheel + CUDA tie-ins
onnxruntime==1.17.*
# Explicitly block the GPU package via constraints
onnxruntime-gpu==0.0.0
EOF

# ---- Build a filtered requirements file (robust AWK) ----
TMP_REQ=$(mktemp)
awk '
  /^[[:space:]]*#/ { next }          # drop comments
  /^[[:space:]]*$/ { next }          # drop blanks
  /^[[:space:]]*-r[[:space:]]+/ { next }  # drop nested includes
  match($0,/^[[:space:]]*(chumpy|mediapipe|onnxruntime(-gpu)?|flatbuffers|mmcv-full|insightface)([<>=].*)?$/) { next }
  { print }
' requirementsGithub.txt > "$TMP_REQ"


# ---- Pre-pin resolver to safe versions BEFORE the big install ----
pip install -c "$CONSTR" "flatbuffers>=2,<3" "onnxruntime==1.17.*"

# ---- Install the core requirements with constraints applied ----
pip install -c "$CONSTR" -r "$TMP_REQ"

# ---- (Optional) If some dependency actually needs chumpy, try this:
# pip install --no-build-isolation chumpy==0.70
# or:
# pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git@master

# ---- PyTorch3D pin (as you had) ----
#pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2
# check CUDA toolkits present
#pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt210/download.html

# # Bring NumPy back to a modern 1.x that works with jax/numba/onnxruntime
# pip install --upgrade --no-deps "numpy==1.24.4" "pandas==1.5.3"

# # Let pip reconcile the rest now that NumPy is compatible
# pip install --upgrade onnxruntime pywavelets numba
# # ---- Pins you listed ----
# pip install pandas==1.4.2 numpy==1.21.0 scikit-video==1.1.11

echo "Setup complete ✅  (venv: $(python -V))"
