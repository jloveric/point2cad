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

patch_installed_pymesh() {
    local init
    init="$("${PYTHON}" -c "import pymesh, pathlib; print(pathlib.Path(pymesh.__file__).parent / '__init__.py')" 2>/dev/null)" || return 0
    if grep -q 'from numpy.testing import Tester' "${init}"; then
        sed -i '/from numpy.testing import Tester/d' "${init}"
        sed -i '/^test = Tester().test/d' "${init}"
    fi
}

install_pymesh_wheel_if_built() {
    shopt -s nullglob
    local wheels=( "${PYMESH}"/dist/pymesh2*.whl )
    shopt -u nullglob
    if ((${#wheels[@]})); then
        uv pip install --python "${PYTHON}" --reinstall "${wheels[@]}"
        patch_installed_pymesh
    fi
}

uv sync
install_pymesh_wheel_if_built

if ! "${PYTHON}" -c "import pymesh" 2>/dev/null; then
    if [[ ! -d "${PYMESH}/.git" ]]; then
        git clone https://github.com/PyMesh/PyMesh.git "${PYMESH}"
    fi

    cd "${PYMESH}"
    git fetch origin
    git checkout "${PYMESH_COMMIT}"
    git submodule update --init

    apply_pymesh_patches() {
        local draco="${PYMESH}/third_party/draco/src/draco"

        if ! grep -q '#include <cstddef>' "${draco}/core/hash_utils.h"; then
            sed -i '/#include <functional>/i #include <cstddef>' "${draco}/core/hash_utils.h"
        fi

        for rel in \
            io/parser_utils.cc \
            point_cloud/point_cloud.cc \
            compression/mesh/mesh_edgebreaker_decoder_impl.cc; do
            file="${draco}/${rel}"
            if ! grep -q '#include <limits>' "${file}"; then
                sed -i '/#include <algorithm>/a #include <limits>' "${file}"
            fi
        done

        file="${draco}/compression/attributes/kd_tree_attributes_encoder.cc"
        if ! grep -q '#include <limits>' "${file}"; then
            sed -i '/#include "draco\/compression\/attributes\/kd_tree_attributes_shared.h"/a #include <limits>' "${file}"
        fi

        sed -i "43s|cwd=\"/root/PyMesh/docker/patches\"|cwd=\"${PYMESH}/docker/patches\"|" \
            "${PYMESH}/docker/patches/patch_wheel.py"

        # GCC 10+ defaults to -fno-common; mmg uses tentative global defs in headers.
        mmg_cmake="${PYMESH}/third_party/mmg/CMakeLists.txt"
        if ! grep -q '\-fcommon' "${mmg_cmake}"; then
            sed -i 's/SET(CMAKE_C_FLAGS " -Wno-char-subscripts ${CMAKE_C_FLAGS}")/SET(CMAKE_C_FLAGS " -Wno-char-subscripts -fcommon ${CMAKE_C_FLAGS}")/' \
                "${mmg_cmake}"
        fi

        # numpy 2.x removed numpy.testing.Tester, which pymesh imports at load time.
        pymesh_init="${PYMESH}/python/pymesh/__init__.py"
        if grep -q 'from numpy.testing import Tester' "${pymesh_init}"; then
            sed -i '/from numpy.testing import Tester/d' "${pymesh_init}"
            sed -i '/^test = Tester().test/d' "${pymesh_init}"
        fi
    }
    apply_pymesh_patches

    rm -rf third_party/build/draco third_party/build/mmg

    uv pip install --python "${PYTHON}" -r "${PYMESH}/python/requirements.txt"
    "${PYTHON}" setup.py bdist_wheel

    PY_MINOR="$("${PYTHON}" -c 'import sys; print(sys.version_info.minor)')"
    rm -rf "build_3.${PY_MINOR}" third_party/build

    "${PYTHON}" "${PYMESH}/docker/patches/patch_wheel.py" dist/pymesh2*.whl
    uv pip install --python "${PYTHON}" dist/pymesh2*.whl
    patch_installed_pymesh
    cd "${ROOT}"
fi

"${PYTHON}" -c "import pymesh"

cat <<EOF

Setup complete. Run from ${ROOT}:

  uv run python -m point2cad.main

Optional flags:

  uv run python -m point2cad.main --path_in ./assets/abc_00949.xyzc --path_out ./out

EOF
