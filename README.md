# Mimic the Writing Style using LLaMA 3

A fine-tuning implementation that teaches LLaMA 3 to mimic personal writing styles. The model analyzes writing samples and generates text that matches the author's unique characteristics, including vocabulary choices, sentence structures, and tone.

## Features

- **Writing Style Analysis**: Analyzes input text for:
  - Vocabulary richness and common word usage
  - Sentence structure and complexity
  - Parts of speech distribution
  - Readability metrics (Flesch-Kincaid)
  
- **Data Processing**:
  - Supports multiple input formats (TXT, DOC, DOCX, PDF)
  - Converts writing samples into training data
  - Implements custom chat templates for LLaMA 3

- **Model Fine-tuning**:
  - Uses QLoRA/LoRA for efficient fine-tuning
  - Implements 4-bit quantization
  - Includes wandb integration for training monitoring

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- Unsloth
- NLTK
- Various text processing libraries (python-docx, PyPDF2, etc.)

## Setup

1. Mount Google Drive with collab
2. Install required packages by running the cells
3. Upload a writing sample (minimum 3000 words) when asked
4. Run the data preprocessing and fine-tuning steps
5. Generate content using your personalized style

## Usage

1. Prepare your writing sample (minimum 3000 words)
2. Run the notebook cells sequentially
3. Input your prompt when requested
4. Receive AI-generated text matching your writing style

## Model Details

- Base Model: unsloth/llama-3-8b-Instruct
- Training: QLoRA fine-tuning
- Quantization: 4-bit
- Max Sequence Length: 2048 tokens
