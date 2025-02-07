import torch
from unsloth import FastLanguageModel
import os
import datetime

class ModelInference:
    def __init__(self, config):
        self.config = config
        
    def load_model(self):
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="llama-3-8b-Instruct-mimick",
            max_seq_length=self.config.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
        
    def generate_text(self, prompt, style_summary):
        
        model, tokenizer = self.load_model()
        
        inputs = tokenizer(
            [
                f"<|start_header_id|>system<|end_header_id|>{style_summary}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|> {prompt}<|eot_id|>"
            ],
            return_tensors="pt"
        ).to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        response = tokenizer.batch_decode(outputs)[0].split("<|end_header_id|>")[-1]
        
        
        output_dir = "generated_texts"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/generated_text_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response[:-10])
            
        return response[:-10]  