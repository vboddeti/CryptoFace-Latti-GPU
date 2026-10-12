#!/usr/bin/env bash
# Source after installing the documented toolchain; does not use environment modules.
for tool in cmake git go nvcc cuobjdump h5c++ g++ patch; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing build tool: $tool; see README.md" >&2
    return 2
  fi
done
CRYPTOFACE_NTL_PREFIX="${CRYPTOFACE_NTL_SYSROOT:-/}/usr"
if [[ ! -f "$CRYPTOFACE_NTL_PREFIX/include/NTL/RR.h" ]]; then
  echo "Missing NTL headers: $CRYPTOFACE_NTL_PREFIX/include/NTL/RR.h" >&2
  return 2
fi
export CPATH="$CRYPTOFACE_NTL_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CRYPTOFACE_NTL_PREFIX/lib/x86_64-linux-gnu:$CRYPTOFACE_NTL_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CRYPTOFACE_NTL_PREFIX/lib/x86_64-linux-gnu:$CRYPTOFACE_NTL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CRYPTOFACE_FACE_CPU_THREADS="${CRYPTOFACE_FACE_CPU_THREADS:-8}"
unset PYTHONPATH
