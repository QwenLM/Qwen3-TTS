#!/bin/bash
# Master script to run all Phase 1 diagnostics in sequence
# Usage: bash run_phase1_diagnostics.sh

set -e  # Exit on first error

echo "======================================================================"
echo "QWEN3 TTS - PHASE 1 DIAGNOSTICS MASTER SCRIPT"
echo "======================================================================"
echo ""
echo "This will run 5 diagnostic scripts to identify the root cause"
echo "of the transformers 5.x silence bug."
echo ""
echo "Transformers version:"
python -c "import transformers; print(f'  {transformers.__version__}')"
echo ""
echo "======================================================================"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="diag_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Step 1: Environment Check
echo "======================================================================"
echo "STEP 1/5: Environment Check (~5 min, no GPU required)"
echo "======================================================================"
python diag_env_check.py 2>&1 | tee "$RESULTS_DIR/01_env_check.log"
echo ""

# Step 2: Static Analysis
echo "======================================================================"
echo "STEP 2/5: Static Analysis (~10 min, no GPU required)"
echo "======================================================================"
python diag_static_analysis.py 2>&1 | tee "$RESULTS_DIR/02_static_analysis.log"
echo ""

# Step 3: RoPE Tensors
echo "======================================================================"
echo "STEP 3/5: RoPE Tensor Comparison (~1 hour, CPU ok)"
echo "======================================================================"
python diag_rope_tensors.py 2>&1 | tee "$RESULTS_DIR/03_rope_tensors.log"
if [ -f "rope_inv_freq_computed.pt" ]; then
    cp rope_*.pt "$RESULTS_DIR/"
    echo "  ✓ RoPE tensors saved to $RESULTS_DIR/rope_*.pt"
fi
echo ""

# Step 4: Mask Capture
echo "======================================================================"
echo "STEP 4/5: Causal Mask Capture (~1 hour, requires GPU)"
echo "======================================================================"
python diag_causal_mask_capture.py 2>&1 | tee "$RESULTS_DIR/04_mask_capture.log"
if [ -f "mask_captures.json" ]; then
    cp mask_captures.json "$RESULTS_DIR/"
    echo "  ✓ Mask captures saved to $RESULTS_DIR/mask_captures.json"
fi
echo ""

# Step 5: Position IDs Trace
echo "======================================================================"
echo "STEP 5/5: Position IDs and cache_position Trace (~1 hour, requires GPU)"
echo "======================================================================"
python diag_position_ids.py 2>&1 | tee "$RESULTS_DIR/05_position_ids.log"
if [ -f "position_trace.json" ]; then
    cp position_trace.json "$RESULTS_DIR/"
    echo "  ✓ Position trace saved to $RESULTS_DIR/position_trace.json"
fi
echo ""

echo "======================================================================"
echo "PHASE 1 DIAGNOSTICS COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "1. Archive baseline (if transformers 4.x):"
echo "   mkdir rope_4x_backup && cp $RESULTS_DIR/rope_*.pt rope_4x_backup/"
echo "   cp $RESULTS_DIR/mask_captures.json mask_captures_4x.json"
echo "   cp $RESULTS_DIR/position_trace.json position_trace_4x.json"
echo ""
echo "2. Switch to transformers 5.x, run this script again"
echo ""
echo "3. Compare outputs using the decision tree in DIAGNOSTIC_PLAN.md"
echo ""
echo "See DIAGNOSTIC_PLAN.md for detailed analysis instructions."
