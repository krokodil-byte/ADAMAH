#!/bin/bash
# Compile GLSL compute shaders to SPIR-V

SHADER_DIR="$(dirname "$0")"
cd "$SHADER_DIR"

# Check if shader compiler is available
if command -v glslc &> /dev/null; then
    COMPILER="glslc"
elif command -v glslangValidator &> /dev/null; then
    COMPILER="glslangValidator"
else
    echo "Error: No shader compiler found. Install Vulkan SDK or glslang-tools."
    echo "  Ubuntu: sudo apt install glslang-tools vulkan-tools"
    echo "  Or download from: https://vulkan.lunarg.com/sdk/home"
    exit 1
fi

echo "Using compiler: $COMPILER"
echo "Compiling shaders..."

# Compile each shader
for shader in unary binary scalar; do
    echo "  $shader.comp -> $shader.spv"

    if [ "$COMPILER" = "glslc" ]; then
        glslc -fshader-stage=compute "$shader.comp" -o "$shader.spv"
    else
        glslangValidator -V "$shader.comp" -o "$shader.spv" --quiet
    fi

    if [ $? -ne 0 ]; then
        echo "Error compiling $shader.comp"
        exit 1
    fi
done

echo "Done! All shaders compiled successfully."

# Show file sizes
echo ""
echo "Compiled SPIR-V sizes:"
ls -lh *.spv | awk '{print "  " $9 ": " $5}'
