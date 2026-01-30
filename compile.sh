#!/bin/bash
set -e

# Compila nella directory dello script
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════════"
echo "           ADAMAH - Build Script"
echo "════════════════════════════════════════════════════════════"

# Step 1: Pulisci vecchi .spv
echo ""
echo "[1/3] Cleaning old .spv files..."
rm -f adamah/shaders/*.spv
echo "      Done."

# Step 2: Compila shaders
echo ""
echo "[2/3] Compiling shaders..."
for comp in adamah/shaders/*.comp; do
    name=$(basename "$comp" .comp)
    glslangValidator -V "$comp" -o "adamah/shaders/${name}.spv" --quiet
    echo "      $name.comp → $name.spv ✓"
done

# Step 3: Compila libreria C
echo ""
echo "[3/3] Compiling C library..."
gcc -shared -fPIC -O3 -o adamah/adamah.so adamah/adamah.c -lvulkan
echo "      adamah.c → adamah.so ✓"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "           ✅ Build complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Run tests with:  python tests/test_all_ops.py"
