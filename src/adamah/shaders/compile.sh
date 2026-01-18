#!/bin/bash
# Compile ADAMAH v4 shaders

GLSLC="glslangValidator"

echo "Compiling ADAMAH v4 shaders..."

$GLSLC -V map_op1.comp -o map_op1.spv && echo "  map_op1.spv OK"
$GLSLC -V map_op2.comp -o map_op2.spv && echo "  map_op2.spv OK"

echo "Done!"
