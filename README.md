# Mimic the Writing Style using LLaMA 3

A fine-tuning implementation that teaches LLaMA 3 to mimic personal writing styles. The model analyzes writing samples and generates text that matches the author's unique writing characteristics such as vocabulary choices, sentence structures, and tone.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Writing Style Analysis**: Analyzes the input text for:
  - Vocabulary richness and common word usage
  - Sentence structure and complexity
  - Parts of speech distribution
  - Readability metrics (Flesch-Kincaid)

- **Data Processing**:
  - Supports multiple input formats (TXT, DOC, DOCX, PDF)
  - Converts writing samples into formatted training data
  - Implements custom chat templates for LLaMA 3

- **Model Fine-tuning**:
  - Uses QLoRA/LoRA for efficient fine-tuning
  - Implements 4-bit quantization for reduced memory usage
  - Integrates WandB for training monitoring

- **Evaluation**:
  - Evaluates style similarity using multiple metrics (vocabulary, POS, sentence length, n-gram overlaps)
  - Computes model perplexity
  - Compares baseline and fine-tuned models

## Installation

### Prerequisites

- Python 3.9+
- PyTorch
- Transformers
- Unsloth
- NLTK
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM
- At least 8GB free disk space for model weights

### Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Setup


1. **Running on Google Colab**:  
   If you're using the provided Jupyter Notebook (or Google Colab) environment, follow the steps in the `Run_in_Notebook.ipynb` file

2. **Running on Desktop**:  
   If you're running the project on your desktop via the terminal, install the latest version of Unsloth directly from its GitHub repository and follow the steps below:

   ```bash
   pip install "unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git"
   ```

## Usage

### Training

1. **Prepare Your Writing Sample**:  
   Ensure you have a writing sample (minimum 3000 words) in one of the supported formats (TXT, DOC, DOCX, PDF).

2. **Preprocessing and Training**:
   Run the following command to preprocess your data and fine-tune the model:
   
   ```bash
   python main.py --mode train --input_file path/to/sample.txt
   ```
   
   This command will:
   - Analyze the writing style and create a style summary.
   - Prepare the training dataset.
   - Fine-tune the LLaMA 3 model using QLoRA with 4-bit quantization.
   
3. **Monitoring**:  
   Check the console logs and the WandB dashboard for training metrics such as loss, learning rate, and accuracy.

### Inference

Generate text in your personalized writing style by providing a prompt:

```bash
python main.py --mode inference --prompt "Your prompt text here"
```

The generated text is printed to the console and saved in the `generated_texts` folder with a timestamped filename.

### Evaluation

The project includes an evaluation module that measures:

- **Style Similarity**: Compares the original writing style with the generated text.
- **Perplexity**: Computes a perplexity score to assess model performance.
- **Baseline Comparison**: Compares output from the base and fine-tuned models.

To run the evaluation script, use:

```bash
python src/evaluation/evaluate.py --original path/to/original.txt --generated path/to/generated.txt --prompt "Your prompt" --output evaluation_results/result.json
```

Examine the output JSON file for detailed metrics on style similarity, perplexity, and comparison between the models.

## Model Details

- **Base Model**: unsloth/llama-3-8b-Instruct
- **Training Method**: QLoRA fine-tuning
- **Quantization**: 4-bit for a lower memory footprint
- **Max Sequence Length**: 2048 tokens

## API Reference

- **DataPreprocessor**  
  Prepares and analyzes input documents, extracting a style summary.
  - `analyze_writing_style(text: str)`: Generates a style summary for the provided text.

- **ModelTrainer**  
  Handles the fine-tuning of LLaMA 3 using custom datasets.
  
- **ModelInference**  
  Loads the fine-tuned model to generate text based on a prompt.
  - `generate_text(prompt, style_summary)`: Produces text matching the writing style.

- **Evaluation Modules**  
  - **StyleEvaluator**: Computes similarity metrics (vocabulary, POS, sentence length, n-grams).
  - **PerplexityEvaluator**: Calculates model perplexity.
  - **BaselineComparator**: Compares performance between the base and fine-tuned models.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LLaMA Team**: For the cutting-edge base model.
- **Unsloth**: For the optimization tools.
- **Open Source Community**: For invaluable contributions and feedback.


