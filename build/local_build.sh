#!/usr/bin/env bash
# Local (non-Docker) setup for Point2CAD.
#
# Prerequisites:
#   - uv: https://docs.astral.sh/uv/getting-started/installation/
#   - System packages (Debian/Ubuntu):
#       sudo apt-get install -y \
#           build-essential cmake git patchelf zip unzip \
#           libgmp-dev libmpfr-dev libgmpxx4ldbl \
#           libboost-dev libboost-thread-dev \
#           libspatialindex-dev libgl1-mesa-glx libxrender1
#
# Run from the repository root:
#   bash build/local_build.sh
#
# Then run the pipeline:
#   uv run python -m point2cad.main

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
VENV="${ROOT}/.venv"
PYMESH="${BUILD}/PyMesh"
PYMESH_COMMIT=384ba882
PYTHON_VERSION=3.10

command -v uv >/dev/null 2>&1 || {
    echo "uv is required but not installed. See https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

cd "${ROOT}"

if [[ -x "${VENV}/bin/python" ]]; then
    VENV_PY="$("${VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${VENV_PY}" != "${PYTHON_VERSION}" ]]; then
        echo "Existing .venv uses Python ${VENV_PY}; PyMesh needs ${PYTHON_VERSION}."
        echo "Remove .venv and re-run:  rm -rf .venv && bash build/local_build.sh"
        exit 1
    fi
else
    uv venv "${VENV}" --python "${PYTHON_VERSION}"
fi

export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"

PYTHON="${VENV}/bin/python"

uv sync

if ! "${PYTHON}" -c "import pymesh; pymesh.test()" 2>/dev/null; then
    if [[ ! -d "${PYMESH}/.git" ]]; then
        git clone https://github.com/PyMesh/PyMesh.git "${PYMESH}"
    fi

    cd "${PYMESH}"
    git fetch origin
    git checkout "${PYMESH_COMMIT}"
    git submodule update --init

    sed -i "43s|cwd=\"/root/PyMesh/docker/patches\"|cwd=\"${PYMESH}/docker/patches\"|" \
        "${PYMESH}/docker/patches/patch_wheel.py"

    uv pip install --python "${PYTHON}" -r "${PYMESH}/python/requirements.txt"
    "${PYTHON}" setup.py bdist_wheel

    PY_MINOR="$("${PYTHON}" -c 'import sys; print(sys.version_info.minor)')"
    rm -rf "build_3.${PY_MINOR}" third_party/build

    "${PYTHON}" "${PYMESH}/docker/patches/patch_wheel.py" dist/pymesh2*.whl
    uv pip install --python "${PYTHON}" dist/pymesh2*.whl
    "${PYTHON}" -c "import pymesh; pymesh.test()"
    cd "${ROOT}"
fi

cat <<EOF

Setup complete. Run from ${ROOT}:

  uv run python -m point2cad.main

Optional flags:

  uv run python -m point2cad.main --path_in ./assets/abc_00949.xyzc --path_out ./out

EOF
