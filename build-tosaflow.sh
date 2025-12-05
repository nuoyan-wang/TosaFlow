#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

JOBS=""

while getopts 'j:' opt; do
  case $opt in
    j) JOBS="${OPTARG}";;
  esac
done

CMAKE_GENERATOR="Unix Makefiles"
if command -v ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

TOSAFLOW_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

git submodule sync || true
git submodule update --init --recursive || true

echo ""
echo ">>> Unified LLVM + MLIR + Clang + TosaFlow build..."
echo ""

cd "${TOSAFLOW_DIR}"
mkdir -p build
cd build

if [[ ! -f "CMakeCache.txt" ]]; then
  cmake -G "${CMAKE_GENERATOR}" \
    ../polygeist/llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_EXTERNAL_PROJECTS="tosaflow" \
    -DLLVM_EXTERNAL_TOSAFLOW_SOURCE_DIR="${TOSAFLOW_DIR}" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_PARALLEL_LINK_JOBS="${JOBS:=}" \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
fi

if [[ "${CMAKE_GENERATOR}" == "Ninja" ]]; then
  if [[ -n "${JOBS}" ]]; then
    ninja -j "${JOBS}"
  else
    ninja
  fi
else
  make -j "${JOBS:-$(nproc)}"
fi

echo ""
echo ">>> Build complete."
echo "TosaFlow tools should be in: ${TOSAFLOW_DIR}/build/bin"
echo "For example:"
echo "  ${TOSAFLOW_DIR}/build/bin/tosa-flow-opt --help"
echo "  ${TOSAFLOW_DIR}/build/bin/tosa-flow-opt \\"
echo "      ${TOSAFLOW_DIR}/samples/pytorch/lenet/lenet_tosa.mlir \\"
echo "      --tosa-flow-pipeline -o lenet_tosa_optimized.mlir"
echo ""
