#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path
import logging
import gc
import torch
import traceback
import json
from docx import Document
import PyPDF2
import numpy as np
import datetime
from src.evaluation.style_metrics import StyleEvaluator

from src.evaluation.perplexity_evaluator import PerplexityEvaluator
from src.data.preprocess import DataPreprocessor
from src.evaluation.baseline_comparator import BaselineComparator
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                
                return obj.item() if obj.ndim == 0 else obj.tolist()
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            
            return super(NpEncoder, self).default(obj)
        except Exception:
            
            return str(obj)

def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_style_evaluation(evaluator, original_text, generated_text):
    
    try:
        clear_memory()
        metrics = evaluator.evaluate_style_similarity(original_text, generated_text)
        return metrics
    except Exception as e:
        logger.error(f"Style evaluation failed: {str(e)}")
        return {"error": str(e)}

def run_preprocessing(preprocessor, original_text):
    
    try:
        clear_memory()
        style_summary = preprocessor.analyze_writing_style(original_text)
        return style_summary
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return "Error in preprocessing: " + str(e)

def run_model_comparison(comparator, prompt, style_summary, original_text):
    
    try:
        clear_memory()
        
        base_results = comparator.evaluate_single_model(
            model_name=comparator.model_name,
            prompt=prompt,
            style_summary=style_summary,
            original_text=original_text
        )
        clear_memory()
        
        fine_tuned_results = comparator.evaluate_single_model(
            model_name=comparator.fine_tuned_name,
            prompt=prompt,
            style_summary=style_summary,
            original_text=original_text
        )
        clear_memory()
        
        return {
            "base_model": base_results,
            "fine_tuned_model": fine_tuned_results
        }
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        return {"error": str(e)}

def read_file(file_path: str) -> str:
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    file_extension = file_path.lower().split('.')[-1]
    
    if file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_extension == 'docx':
        doc = Document(file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    elif file_extension == 'pdf':
        text = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return ' '.join(text)
    else:
        raise ValueError(f"Unsupported file format: .{file_extension}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate style similarity')
    parser.add_argument('--original', type=str, required=True, 
                       help='Path to original text file')
    parser.add_argument('--generated', type=str, required=True, 
                       help='Path to generated text file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output path for evaluation results')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt for generating text in the specified style')
    
    args = parser.parse_args()
    
    results = {
        "style_metrics": None,
        "perplexity": None,
        "baseline_comparison": None
    }
    
    try:
        
        if not os.path.exists(args.original):
            raise FileNotFoundError(f"Original file not found: {args.original}")
        if not os.path.exists(args.generated):
            raise FileNotFoundError(f"Generated file not found: {args.generated}")
            
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        original_text = read_file(args.original)
        generated_text = read_file(args.generated)
        
        # 1. Style Metrics Evaluation
        try:
            style_evaluator = StyleEvaluator()
            results["style_metrics"] = style_evaluator.evaluate_style_similarity(
                original_text, generated_text
            )
            logger.info("Style metrics evaluation completed successfully")
        except Exception as e:
            logger.error(f"Style metrics evaluation failed: {str(e)}")
            results["style_metrics"] = {"error": str(e)}
        
        clear_memory()
        
        # 2. Perplexity Evaluation
        try:
            logger.info("Starting perplexity evaluation...")
            model_name = "llama-3-8b-Instruct-mimick"
            
            logger.info(f"Loading model: {model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=512,
                dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
            
            FastLanguageModel.for_inference(model)
            logger.info("Model prepared for inference")
            
            logger.info("Creating PerplexityEvaluator...")
            perplexity_eval = PerplexityEvaluator(model, tokenizer)
            logger.info("Starting perplexity calculation...")
            
            
            if 'results' not in locals():
                results = {}
            
            try:
                perplexity_result = perplexity_eval.calculate_perplexity(generated_text)
                logger.info(f"Perplexity calculation result: {perplexity_result}")
                results["perplexity"] = perplexity_result
            except Exception as calc_error:
                logger.error(f"Perplexity calculation failed: {str(calc_error)}")
                logger.error(traceback.format_exc())
                results["perplexity"] = {"error": str(calc_error)}
            finally:
                
                try:
                    output_path = os.path.abspath(args.output)
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=4, cls=NpEncoder)
                    logger.info(f"Intermediate results written to: {output_path}")
                except Exception as write_error:
                    logger.error(f"Failed to write intermediate results: {str(write_error)}")
            
            del model, tokenizer
            logger.info("Perplexity evaluation completed successfully")
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            results["perplexity"] = {"error": str(e)}
        finally:
            clear_memory()
        
        clear_memory()
        
        # 3. Baseline Comparison
        try:
            comparator = BaselineComparator()
            
            preprocessor = DataPreprocessor()
            style_summary = run_preprocessing(preprocessor, original_text)
            
            results["baseline_comparison"] = run_model_comparison(
                comparator=comparator,
                prompt=args.prompt,
                style_summary=style_summary,
                original_text=original_text
            )
            logger.info("Baseline comparison completed successfully")
        except Exception as e:
            logger.error(f"Baseline comparison failed: {str(e)}")
            results["baseline_comparison"] = {"error": str(e)}
        
        
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Attempting to write results to: {output_path}")
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4, cls=NpEncoder)
            logger.info(f"Successfully wrote results to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write results: {str(e)}")
            logger.error(traceback.format_exc())
            
            fallback_path = os.path.join(os.getcwd(), "evaluation_results_fallback.json")
            with open(fallback_path, 'w') as f:
                json.dump(results, f, indent=4, cls=NpEncoder)
            logger.info(f"Results written to fallback location: {fallback_path}")
        
        print("\nEvaluation Results:")
        for eval_type, result in results.items():
            if result and "error" not in result:
                print(f"\n{eval_type.upper()} EVALUATION:")
                print(json.dumps(result, indent=2))
            else:
                print(f"\n{eval_type.upper()} EVALUATION FAILED:")
                print(result.get("error", "Unknown error"))
                
        success_count = sum(1 for r in results.values() if r and "error" not in r)
        logger.info(f"Completed {success_count}/3 evaluations successfully")
        
    except Exception as e:
        logger.error(f"Critical error in evaluation: {str(e)}")
        
        logger.error(traceback.format_exc())
        with open(args.output, 'w') as f:
            json.dump({"error": str(e), "results": results}, f, indent=4, cls=NpEncoder)
        sys.exit(1)
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()