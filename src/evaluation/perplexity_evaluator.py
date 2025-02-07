from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import torch
from datasets import Dataset
from nltk.tokenize import sent_tokenize
import numpy as np
from typing import Dict, Union, List
import logging
import traceback
import gc
import os

logger = logging.getLogger(__name__)

class PerplexityEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_length = 512
        self.batch_size = 1
        
       
        self.model.eval()
        
    def _clear_memory(self):
        """Helper function to clear GPU memory"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Memory clearing failed: {str(e)}")

    def prepare_held_out_data(self, text: str) -> Dataset:
        """Prepare text data in chunks for evaluation"""
        try:
            logger.info("Starting text chunking...")
            if not text or not isinstance(text, str):
                raise ValueError(f"Invalid text input: {type(text)}")
                
            sentences = sent_tokenize(text)
            logger.info(f"Split text into {len(sentences)} sentences")
            
            chunks: List[str] = []
            current_chunk: List[str] = []
            current_length = 0
            
            for sentence in sentences:
                tokens = self.tokenizer.tokenize(sentence)
                sentence_length = len(tokens)
                
                if current_length + sentence_length > self.max_length - 100:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            if not chunks:
                raise ValueError("No valid chunks created from input text")
                
            logger.info(f"Created {len(chunks)} chunks from text")
            return Dataset.from_dict({"text": chunks})
            
        except Exception as e:
            logger.error(f"Error in prepare_held_out_data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def custom_collate_fn(self, examples):
        return {"text": [example["text"] for example in examples]}

    def calculate_perplexity(self, text: str) -> dict:
        try:
            logger.info("Starting perplexity calculation...")            
            dataset = self.prepare_held_out_data(text)
            
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_collate_fn
            )
            
            total_loss = 0.0
            total_tokens = 0
            
            self.model.eval()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        
                        inputs = self.tokenizer(
                            batch["text"],
                            return_tensors="pt",
                            padding="longest",
                            truncation=True,
                            max_length=self.max_length
                        )
                        
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        num_tokens = inputs["input_ids"].numel()
                        
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                        
                        logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
                    except Exception as batch_error:
                        logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}", exc_info=True)
                        continue
            
            if total_tokens == 0:
                raise ValueError("No tokens were processed in perplexity evaluation")
            
            avg_loss = total_loss / total_tokens
            perplexity = float(np.exp(avg_loss))
            logger.info(f"Finished perplexity calculation, perplexity: {perplexity:.2f}")
            
            return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}
        
        except Exception as e:
            logger.error("Error in calculate_perplexity", exc_info=True)
            raise 