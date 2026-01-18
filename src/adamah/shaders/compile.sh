#!/bin/bash
# Compile ADAMAH v4 shaders

GLSLC="glslangValidator"

echo "Compiling ADAMAH v4 shaders..."

$GLSLC -V map_op1.comp -o map_op1.spv && echo "  map_op1.spv OK"
$GLSLC -V map_op2.comp -o map_op2.spv && echo "  map_op2.spv OK"
$GLSLC -V map_matmul.comp -o map_matmul.spv && echo "  map_matmul.spv OK"
$GLSLC -V map_reduce.comp -o map_reduce.spv && echo "  map_reduce.spv OK"
$GLSLC -V map_broadcast.comp -o map_broadcast.spv && echo "  map_broadcast.spv OK"

echo "Done!"
ls -la *.spv
