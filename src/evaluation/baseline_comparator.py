from typing import Dict, Any, List
from src.evaluation.style_metrics import StyleEvaluator
from src.evaluation.perplexity_evaluator import PerplexityEvaluator
from unsloth import FastLanguageModel
import torch
import json
import datetime
import os

class BaselineComparator:
    def __init__(self):
        self.style_evaluator = StyleEvaluator()
        self.model_name = "unsloth/llama-3-8b-Instruct"
        self.fine_tuned_name = "llama-3-8b-Instruct-mimick"
        
    def load_model(self, model_name):
        """Load model and return it with its tokenizer"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=512,  
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
        
    def generate_text_with_model(self, model, tokenizer, prompt: str, style_summary: str) -> str:
        """Generate text using specified model"""
        inputs = tokenizer(
            [
                f"<|start_header_id|>system<|end_header_id|>{style_summary}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|> {prompt}<|eot_id|>"
            ],
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to("cuda")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                do_sample=True,
                temperature=0.7
            )
            response = tokenizer.batch_decode(outputs)[0].split("<|end_header_id|>")[-1]
        
        return response[:-10]

    def evaluate_single_model(self, model_name, prompt, style_summary, original_text):
        """Evaluate a single model"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            model, tokenizer = self.load_model(model_name)
            generated_text = self.generate_text_with_model(model, tokenizer, prompt, style_summary)
            
            style_metrics = self.style_evaluator.evaluate_style_similarity(
                original_text, generated_text
            )
            
            perplexity = PerplexityEvaluator(model, tokenizer).calculate_perplexity(generated_text)
            
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return {
                "generated_text": generated_text,
                "style_metrics": style_metrics,
                "perplexity": perplexity
            }
        except Exception as e:
            return {"error": str(e)}

    def compare_models(self, prompt: str, style_summary: str, original_text: str) -> Dict[str, Any]:
        """Compare base and fine-tuned models"""
        comparison = {"base_model": {}, "fine_tuned_model": {}}
        
        # Evaluate base model
        base_results = self.evaluate_single_model(
            self.model_name, prompt, style_summary, original_text
        )
        comparison["base_model"] = base_results
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        fine_tuned_results = self.evaluate_single_model(
            self.fine_tuned_name, prompt, style_summary, original_text
        )
        comparison["fine_tuned_model"] = fine_tuned_results
        if "error" not in base_results and "error" not in fine_tuned_results:
            comparison["improvements"] = {
                "perplexity_improvement": (
                    base_results["perplexity"]["perplexity"] - 
                    fine_tuned_results["perplexity"]["perplexity"]
                ),
                "style_metrics_improvement": {
                    metric: fine_tuned_results["style_metrics"][metric] - 
                            base_results["style_metrics"][metric]
                    for metric in fine_tuned_results["style_metrics"]
                }
            }
        
        self._save_comparison_results(comparison)
        return comparison
    
    def _save_comparison_results(self, comparison: Dict[str, Any]) -> None:
        """Save comparison results to file"""
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/model_comparison_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=4) 