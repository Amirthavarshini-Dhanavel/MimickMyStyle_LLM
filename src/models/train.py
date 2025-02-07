import torch
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
import wandb
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def setup_model(self):
        
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model, tokenizer
        
    def train(self, dataset):
       
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )        
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.15,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=True,
            use_dora=True,
            fan_in_fan_out=False,
            init_lora_weights=True,
        )        
        
        def format_conversations(example):
            formatted_text = ""
            for message in example['conversations']:
                if message['role'] == 'system':
                    formatted_text += f"<|start_header_id|>system<|end_header_id|>{message['content']}<|eot_id|>"
                elif message['role'] == 'user':
                    formatted_text += f"<|start_header_id|>user<|end_header_id|>{message['content']}<|eot_id|>"
                elif message['role'] == 'assistant':
                    formatted_text += f"<|start_header_id|>assistant<|end_header_id|>{message['content']}<|eot_id|>"
            return {"formatted_text": formatted_text}        
        
        formatted_dataset = dataset.map(format_conversations)        
        
        split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']        
        
        training_args = TrainingArguments(
            output_dir="outputs",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_ratio=0.15,
            num_train_epochs=5,
            learning_rate=5e-5,
            lr_scheduler_type="cosine_with_restarts",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.005,
            max_grad_norm=0.5,
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="loss"
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="formatted_text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args=training_args
        )       
        
        wandb.init(project="llama-3-8b-Instruct-mimick")
        trained_model = trainer.train()     
        
        model.save_pretrained("llama-3-8b-Instruct-mimick")
        tokenizer.save_pretrained("llama-3-8b-Instruct-mimick")
        
        return trained_model