#!/bin/bash
# ============================================================================
# ADAMAH v5.1.0 — Installer
#
# Builds adamah.so from source, compiles shaders (optional), and installs
# the library system-wide so you can `import adamah` from anywhere.
#
# Usage:
#   ./install.sh              # install to /opt/adamah (needs sudo)
#   ./install.sh --prefix ~   # install to ~/adamah (no sudo needed)
#   ./install.sh --help
#
# After install you can delete this source directory.
# ============================================================================
set -e

VERSION="5.1.0"
DEFAULT_PREFIX="/opt/adamah"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse args ──────────────────────────────────────────────────────────────
PREFIX=""
SKIP_SHADERS=0
VERBOSE=0

print_help() {
    echo "ADAMAH v${VERSION} Installer"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --prefix DIR    Install to DIR (default: ${DEFAULT_PREFIX})"
    echo "  --skip-shaders  Don't recompile shaders (use precompiled .spv)"
    echo "  --verbose       Show compiler output"
    echo "  --help          This message"
    echo ""
    echo "Requirements:"
    echo "  - gcc"
    echo "  - Vulkan SDK (libvulkan-dev)"
    echo "  - Python 3.8+ with numpy"
    echo "  - glslangValidator (optional, for shader recompilation)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)    PREFIX="$2"; shift 2;;
        --skip-shaders) SKIP_SHADERS=1; shift;;
        --verbose)   VERBOSE=1; shift;;
        --help|-h)   print_help; exit 0;;
        *)           echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

[ -z "$PREFIX" ] && PREFIX="$DEFAULT_PREFIX"
INSTALL_DIR="$PREFIX"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
step() { echo -e "\n${CYAN}${BOLD}[$1]${NC} $2"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        ADAMAH v${VERSION} — Installer         ║"
echo "  ║   Vulkan GPU Compute · CUDA Alternative  ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Check dependencies ──────────────────────────────────────────────────
step "1/5" "Checking dependencies"

# gcc
if command -v gcc &>/dev/null; then
    ok "gcc $(gcc -dumpversion)"
else
    fail "gcc not found. Install with: sudo apt install build-essential"
fi

# Vulkan
VULKAN_OK=0
if pkg-config --exists vulkan 2>/dev/null; then
    ok "Vulkan SDK (pkg-config)"
    VULKAN_OK=1
elif [ -f /usr/include/vulkan/vulkan.h ]; then
    ok "Vulkan headers found"
    VULKAN_OK=1
elif ldconfig -p 2>/dev/null | grep -q libvulkan; then
    ok "libvulkan found"
    VULKAN_OK=1
fi
if [ $VULKAN_OK -eq 0 ]; then
    fail "Vulkan not found. Install with: sudo apt install libvulkan-dev vulkan-tools"
fi

# Python
PYTHON=""
for p in python3 python; do
    if command -v "$p" &>/dev/null; then
        PY_VER=$("$p" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        PY_MAJ=$("$p" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        PY_MIN=$("$p" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$PY_MAJ" -ge 3 ] && [ "$PY_MIN" -ge 8 ] 2>/dev/null; then
            PYTHON="$p"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    fail "Python 3.8+ not found"
fi
ok "Python ${PY_VER} (${PYTHON})"

# numpy
if $PYTHON -c "import numpy" 2>/dev/null; then
    NP_VER=$($PYTHON -c "import numpy; print(numpy.__version__)")
    ok "numpy ${NP_VER}"
else
    warn "numpy not found — installing..."
    $PYTHON -m pip install numpy --break-system-packages 2>/dev/null || \
    $PYTHON -m pip install numpy 2>/dev/null || \
    fail "Could not install numpy. Run: $PYTHON -m pip install numpy"
    ok "numpy installed"
fi

# glslangValidator (optional)
HAS_GLSLANG=0
if command -v glslangValidator &>/dev/null; then
    ok "glslangValidator (shader compiler)"
    HAS_GLSLANG=1
else
    warn "glslangValidator not found — will use precompiled shaders"
    warn "To recompile: sudo apt install glslang-tools"
fi

# ── 2. Compile shaders (if glslang available) ──────────────────────────────
step "2/5" "Compiling shaders"

SHADER_SRC="$SCRIPT_DIR/adamah/shaders/src"
SHADER_OUT="$SCRIPT_DIR/adamah/shaders"
SHADERS_COMPILED=0

if [ $SKIP_SHADERS -eq 1 ]; then
    warn "Skipped (--skip-shaders)"
elif [ $HAS_GLSLANG -eq 1 ] && [ -d "$SHADER_SRC" ]; then
    for dtype in f32 bf16 q8; do
        src_dir="$SHADER_SRC/$dtype"
        out_dir="$SHADER_OUT/$dtype"
        [ -d "$src_dir" ] || continue
        mkdir -p "$out_dir"
        count=0
        for comp in "$src_dir"/*.comp; do
            [ -f "$comp" ] || continue
            name=$(basename "${comp%.comp}")
            out="$out_dir/${name}.spv"
            if [ $VERBOSE -eq 1 ]; then
                glslangValidator -V "$comp" -o "$out"
            else
                glslangValidator -V "$comp" -o "$out" >/dev/null 2>&1
            fi
            count=$((count + 1))
        done
        ok "$dtype: $count shaders compiled"
        SHADERS_COMPILED=$((SHADERS_COMPILED + count))
    done
    # Copy f32 to root for backward compat
    cp "$SHADER_OUT/f32"/*.spv "$SHADER_OUT/" 2>/dev/null || true
    ok "Total: $SHADERS_COMPILED shaders"
else
    # Check precompiled exist
    if [ -f "$SHADER_OUT/f32/map_op1.spv" ]; then
        ok "Using precompiled .spv shaders"
    else
        fail "No shader sources or precompiled .spv found"
    fi
fi

# ── 3. Build adamah.so ─────────────────────────────────────────────────────
step "3/5" "Building adamah.so"

# The SHADER_PATH define tells the .so where to find shaders at runtime
# We point it to the install directory
RUNTIME_SHADER_PATH="${INSTALL_DIR}/adamah/shaders"

cd "$SCRIPT_DIR/adamah"

if [ $VERBOSE -eq 1 ]; then
    echo "  gcc -shared -fPIC -O2 -march=native -DSHADER_PATH=\"${RUNTIME_SHADER_PATH}\" adamah.c -o adamah.so -lvulkan -ldl -lm"
fi

gcc -shared -fPIC -O2 -march=native \
    -DSHADER_PATH="\"${RUNTIME_SHADER_PATH}\"" \
    adamah.c -o adamah.so \
    -lvulkan -ldl -lm

if [ ! -f adamah.so ]; then
    fail "Build failed"
fi

SO_SIZE=$(du -h adamah.so | cut -f1)
ok "adamah.so built (${SO_SIZE})"

# Verify SHADER_PATH is baked in
if strings adamah.so | grep -q "${RUNTIME_SHADER_PATH}"; then
    ok "SHADER_PATH = ${RUNTIME_SHADER_PATH}"
else
    fail "SHADER_PATH not found in adamah.so — compiler define failed"
fi

cd "$SCRIPT_DIR"

# ── 4. Install ──────────────────────────────────────────────────────────────
step "4/5" "Installing to ${INSTALL_DIR}"

NEED_SUDO=0
if [ ! -w "$(dirname "$INSTALL_DIR")" ] 2>/dev/null; then
    NEED_SUDO=1
fi

do_install() {
    local SUDO=""
    [ $NEED_SUDO -eq 1 ] && SUDO="sudo"

    # Clean old installation if present
    if [ -d "$INSTALL_DIR" ]; then
        warn "Removing previous installation at $INSTALL_DIR"
        $SUDO rm -rf "$INSTALL_DIR"
    fi

    # Create install directory
    $SUDO mkdir -p "$INSTALL_DIR"

    # Copy everything
    $SUDO cp -r "$SCRIPT_DIR/adamah"    "$INSTALL_DIR/"
    $SUDO cp -r "$SCRIPT_DIR/benchmarks" "$INSTALL_DIR/" 2>/dev/null || true
    $SUDO cp -r "$SCRIPT_DIR/tests"      "$INSTALL_DIR/" 2>/dev/null || true
    $SUDO cp    "$SCRIPT_DIR/LICENSE"     "$INSTALL_DIR/" 2>/dev/null || true
    $SUDO cp    "$SCRIPT_DIR/pyproject.toml" "$INSTALL_DIR/" 2>/dev/null || true

    # Create uninstall script in install dir
    $SUDO tee "$INSTALL_DIR/uninstall.sh" > /dev/null << 'UNINSTALL'
#!/bin/bash
set -e
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
PTH_FILE=""
for sp in $(python3 -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null); do
    [ -f "$sp/adamah.pth" ] && PTH_FILE="$sp/adamah.pth"
done
echo "Uninstalling ADAMAH from $INSTALL_DIR"
[ -n "$PTH_FILE" ] && sudo rm -f "$PTH_FILE" && echo "  Removed $PTH_FILE"
sudo rm -rf "$INSTALL_DIR"
echo "  Done."
UNINSTALL
    $SUDO chmod +x "$INSTALL_DIR/uninstall.sh"

    # Create .pth file so Python finds adamah globally
    # This adds INSTALL_DIR to sys.path for all Python invocations
    local PTH_INSTALLED=0
    for site_dir in $($PYTHON -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null); do
        if [ -d "$site_dir" ]; then
            $SUDO sh -c "echo '$INSTALL_DIR' > '$site_dir/adamah.pth'"
            ok "Python path: $site_dir/adamah.pth"
            PTH_INSTALLED=1
            break
        fi
    done

    if [ $PTH_INSTALLED -eq 0 ]; then
        # Fallback: user site-packages
        local user_site=$($PYTHON -c "import site; print(site.getusersitepackages())")
        mkdir -p "$user_site"
        echo "$INSTALL_DIR" > "$user_site/adamah.pth"
        ok "Python path (user): $user_site/adamah.pth"
    fi
}

if [ $NEED_SUDO -eq 1 ]; then
    echo -e "  ${YELLOW}Requires sudo for ${INSTALL_DIR}${NC}"
fi
do_install
ok "Files installed"

# ── 5. Verify ───────────────────────────────────────────────────────────────
step "5/5" "Verifying installation"

# Test import from a temp directory (not the source tree)
cd /tmp
IMPORT_OK=$($PYTHON -c "
import adamah
gpu = adamah.Adamah()
# Test f32 init
dtype = gpu.get_dtype()
# Test bf16 shader loading
gpu.set_dtype(1)  # DTYPE_BF16
gpu.set_dtype(0)  # back to f32
print(f'OK adamah v{adamah.__version__ if hasattr(adamah, \"__version__\") else \"?\"}  [f32+bf16 shaders verified]')
" 2>&1) || true

if echo "$IMPORT_OK" | grep -q "OK"; then
    ok "$IMPORT_OK"
else
    warn "Import test issue: $IMPORT_OK"
    warn "You may need to restart your shell or set PYTHONPATH=$INSTALL_DIR"
fi

cd "$SCRIPT_DIR"

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ╔══════════════════════════════════════════╗"
echo "  ║          Installation complete!           ║"
echo -e "  ╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Usage:"
echo "    python3 -c \"import adamah; gpu = adamah.init(); print('Ready')\""
echo ""
echo "  Benchmarks:"
echo "    python3 ${INSTALL_DIR}/benchmarks/benchmark_simple_batches.py"
echo "    python3 ${INSTALL_DIR}/benchmarks/benchmark_simple_batches.py --dtype bf16"
echo "    python3 ${INSTALL_DIR}/benchmarks/benchmark_simple_batches.py --dtype q8"
echo ""
echo "  Uninstall:"
echo "    ${INSTALL_DIR}/uninstall.sh"
echo ""
