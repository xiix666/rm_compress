#!/usr/bin/env bash
set -euo pipefail
MODELS_DIR="$(cd "$(dirname "$0")/../models" && pwd)"

for onnx_file in "$MODELS_DIR"/*.onnx; do
    name=$(basename "$onnx_file" .onnx)
    echo "Converting $onnx_file -> $MODELS_DIR/$name.xml"
    ovc "$onnx_file" --output_model "$MODELS_DIR/$name.xml" --compress_to_fp16=True
done

for onnx_file in "$MODELS_DIR"/*h_s*.onnx; do
    [[ -e "$onnx_file" ]] || continue
    name=$(basename "$onnx_file" .onnx)
    echo "Converting $onnx_file -> $MODELS_DIR/${name}_fp32.xml"
    ovc "$onnx_file" --output_model "$MODELS_DIR/${name}_fp32.xml" --compress_to_fp16=False
done

echo "All conversions done."
ls -lh "$MODELS_DIR"/*.xml "$MODELS_DIR"/*.bin
