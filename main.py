from config.config import ModelConfig
from src.data.preprocess import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.inference import ModelInference
import json
import argparse
import os
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--input_file", help="Path to input file for training")
    parser.add_argument("--prompt", help="Prompt for inference")
    args = parser.parse_args()

    config = ModelConfig()
    
    # Create necessary directories
    os.makedirs("./data", exist_ok=True)

    if args.mode == "train":
        # Preprocess data
        preprocessor = DataPreprocessor()
        style_summary = preprocessor.prepare_dataset(args.input_file, "./data")
        
        # Load and format dataset
        dataset = load_from_disk("./data/preprocessed_data")
        
        # Format conversations
        def format_conversations(example):
            formatted_text = ""
            for message in example['conversations']:
                if message['role'] == 'system':
                    formatted_text += f"<|system|>{message['content']}<|end|>"
                elif message['role'] == 'user':
                    formatted_text += f"<|user|>{message['content']}<|end|>"
                elif message['role'] == 'assistant':
                    formatted_text += f"<|assistant|>{message['content']}<|end|>"
            return {"text": formatted_text}
        
        formatted_dataset = dataset.map(format_conversations)
        
        # Train model with formatted dataset
        trainer = ModelTrainer(config)
        trained_model = trainer.train(dataset=formatted_dataset)
        
        # Save training stats
        with open("trainer_stats.json", "w") as f:
            json.dump(trained_model, f, indent=4)
            
    elif args.mode == "inference":
        # Load style summary
        with open("./data/style_summary.txt", "r") as f:
            style_summary = f.read()
            
        # Generate text
        inference = ModelInference(config)
        generated_text = inference.generate_text(args.prompt, style_summary)
        print(generated_text)

if __name__ == "__main__":
    main()