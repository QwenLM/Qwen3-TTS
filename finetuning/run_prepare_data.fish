#!/usr/bin/env fish
# SLURM job array script for running prepare_data.py in parallel
#
# Usage: ./run_prepare_data.fish <input_jsonl> <output_jsonl> [num_shards] [partition]
#
# Example:
#   ./run_prepare_data.fish /path/to/input.jsonl /path/to/output.jsonl 16 bw2_lowprio

# Default configuration
set -l DEFAULT_NUM_SHARDS 8
set -l DEFAULT_PARTITION "bw2_lowprio"

# Parse arguments
if test (count $argv) -lt 2
    echo "Usage: $argv[0] <input_jsonl> <output_jsonl> [num_shards] [partition]"
    echo ""
    echo "Arguments:"
    echo "  input_jsonl   - Path to input JSONL file"
    echo "  output_jsonl  - Path to output JSONL file (shard suffix will be added)"
    echo "  num_shards    - Number of parallel jobs (default: $DEFAULT_NUM_SHARDS)"
    echo "  partition     - SLURM partition (default: $DEFAULT_PARTITION)"
    exit 1
end

set -l INPUT_JSONL $argv[1]
set -l OUTPUT_JSONL $argv[2]
set -l NUM_SHARDS (test (count $argv) -ge 3; and echo $argv[3]; or echo $DEFAULT_NUM_SHARDS)
set -l PARTITION (test (count $argv) -ge 4; and echo $argv[4]; or echo $DEFAULT_PARTITION)

# Validate input file exists
if not test -f "$INPUT_JSONL"
    echo "Error: Input file '$INPUT_JSONL' does not exist"
    exit 1
end

# Get script directory (where prepare_data.py is located)
set -l SCRIPT_DIR (dirname (status --current-filename))
set -l PREPARE_DATA_SCRIPT "$SCRIPT_DIR/prepare_data.py"

if not test -f "$PREPARE_DATA_SCRIPT"
    echo "Error: prepare_data.py not found at '$PREPARE_DATA_SCRIPT'"
    exit 1
end

# Get workspace root (for activating the virtual environment)
set -l WORKSPACE_ROOT (dirname $SCRIPT_DIR)
set -l VENV_PATH "$WORKSPACE_ROOT/env"

# Create output directory if it doesn't exist
set -l OUTPUT_DIR (dirname $OUTPUT_JSONL)
mkdir -p "$OUTPUT_DIR"

# Create logs directory
set -l LOGS_DIR "$OUTPUT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Create a temporary SLURM batch script
set -l SLURM_SCRIPT (mktemp /tmp/slurm_prepare_data_XXXXXX.sh)

echo "#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --array=0-"(math $NUM_SHARDS - 1)"
#SBATCH --partition=$PARTITION
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu=8
#SBATCH --output=$LOGS_DIR/prepare_data_%A_%a.out
#SBATCH --error=$LOGS_DIR/prepare_data_%A_%a.err

# Activate virtual environment if it exists
if [ -d \"$VENV_PATH\" ]; then
    source \"$VENV_PATH/bin/activate\"
fi

# Get shard ID from SLURM array task ID
SHARD_ID=\$SLURM_ARRAY_TASK_ID

echo \"Starting shard \$SHARD_ID of $NUM_SHARDS\"
echo \"Input: $INPUT_JSONL\"
echo \"Output: $OUTPUT_JSONL\"
echo \"\"

# Run the prepare_data.py script
python \"$PREPARE_DATA_SCRIPT\" \\
    --input_jsonl \"$INPUT_JSONL\" \\
    --output_jsonl \"$OUTPUT_JSONL\" \\
    --shard_id \$SHARD_ID \\
    --num_shards $NUM_SHARDS \\
    --batch_size 32 \\
    --tokenizer_model_path \"Qwen/Qwen3-TTS-Tokenizer-12Hz\" \\
    --device cuda

echo \"\"
echo \"Shard \$SHARD_ID completed with exit code \$?\"
" > $SLURM_SCRIPT

echo "Created SLURM script: $SLURM_SCRIPT"
echo ""
echo "Configuration:"
echo "  Input:      $INPUT_JSONL"
echo "  Output:     $OUTPUT_JSONL"
echo "  Shards:     $NUM_SHARDS"
echo "  Partition:  $PARTITION"
echo "  Logs:       $LOGS_DIR"
echo ""

# Submit the job array
echo "Submitting SLURM job array..."
set -l JOB_ID (sbatch --parsable $SLURM_SCRIPT)

if test $status -eq 0
    echo "Job array submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "After completion, merge shards with:"
    echo "  cat $OUTPUT_JSONL.shard*_of_* > $OUTPUT_JSONL"
else
    echo "Error: Failed to submit job"
    exit 1
end

# Clean up the temporary script after a delay (give SLURM time to read it)
# Or keep it for debugging - uncomment below to auto-remove
# sleep 5 && rm -f $SLURM_SCRIPT &
