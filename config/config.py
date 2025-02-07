from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "unsloth/llama-3-8b-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    learning_rate: float = 1e-5