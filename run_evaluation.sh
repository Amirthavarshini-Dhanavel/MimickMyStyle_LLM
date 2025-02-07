#!/bin/bash

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# Additional memory optimization for PyTorch
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Set Python path (adjust this to your project root)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create directories if they don't exist and ensure proper permissions
mkdir -p evaluation_results
chmod 777 evaluation_results
mkdir -p logs
chmod 777 logs

# Get current timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Clear GPU cache before starting
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache()"

# Run model comparison
echo "Running model comparison..."
python src/evaluation/evaluate.py \
    --original "Mywritingsample.docx" \
    --generated "generated_texts/generated_text_20250206_034942.txt" \
    --output "evaluation_results/model_comparison_${timestamp}.json" \
    --prompt "Write a letter to my manager stating that I'll not be able to come to office for next 3 days. Since I am extremely sick and have fever" \
    2>&1 | tee "logs/model_comparison_${timestamp}.log"

# After running evaluate.py, verify the output file exists
output_file="evaluation_results/model_comparison_${timestamp}.json"
log_file="logs/model_comparison_${timestamp}.log"

if [ -f "$output_file" ]; then
    echo "Results file created successfully at: $output_file"
    echo "File contents:"
    cat "$output_file"
else
    echo "Error: Results file was not created at: $output_file"
    echo "Log file contents:"
    cat "$log_file"
fi

# Final cleanup
python -c "import torch; torch.cuda.empty_cache()" 