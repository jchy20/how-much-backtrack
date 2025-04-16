from dataclasses import dataclass
import torch
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

@dataclass
class TaskActivations:
    task_name: str
    attention_activations: Dict[str, torch.Tensor]


class ActivationManager:
    def __init__(self, model, model_tokenizer, device):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.device = device
        self.attention_activations = {}     
        
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_heads = self.model.config.num_key_value_heads
        
        self._register_hooks()
            
    # def compute_activations(self, query: str, context: str, gold_information: List[str]) -> TaskActivations:
    #     """Compute activations with character-level alignment"""                
    #     self.attention_activations.clear()
        
    #     try:
    #         # Prepare full input text
    #         model_input_text = prepare_sufficiency_input(query, context)
    
    #         model_inputs = self.model_tokenizer(
    #             model_input_text,
    #             return_tensors="pt",
    #             padding=True,
    #         ).to(self.device)
            
    #         # Find gold location
    #         gold_location = self.find_gold_location(model_input_text, gold_information)
    #         if gold_location is None:
    #             print("Gold location not found")
            
    #         # Compute activations
    #         with torch.no_grad():
    #             outputs = self.model(**model_inputs)
                            
    #             # Clear outputs immediately as they're not needed
    #             del outputs
            
    #         attention_activations = self._process_head_features()
            
    #         context_activations = ContextActivations(
    #             full_context=model_input_text,
    #             model_token_ids=model_inputs.input_ids.cpu(),
    #             attention_activations=attention_activations,
    #             gold_location=gold_location,
    #         )
    #         del model_inputs, attention_activations
    #         torch.cuda.empty_cache()

    #         return context_activations
            
    #     except Exception as e:
    #         print(f"Error computing activations: {str(e)}")
    #         raise

    def _process_head_features(self) -> Dict[str, torch.Tensor]:
        """Process attention activations into head features, preserving position information"""
        head_features = {}
        for layer_name, activation in self.attention_activations.items():
            if activation is not None:
                # No need to access [-1] since we're storing single activations
                head_features[layer_name] = activation[0]   
        return head_features
        

    def _register_hooks(self):
        """Register forward hooks to capture attention layer outputs"""
        def get_activation(name):
            def hook(module, input, output):
                processed = self._qwen2_process_activation(output)
                if processed is not None:
                    layer_num = name.split('.layers.')[1].split('.')[0]
                    hook_name = f"layer_{layer_num}"
                    self.attention_activations[hook_name]=processed
            return hook

        for name, module in self.model.named_modules():
            if "self_attn.o_proj" in name:
                module.register_forward_hook(get_activation(name))

    def _validate_and_sanitize_activation(self, activation: torch.Tensor, context: str = "") -> Optional[torch.Tensor]:
        """Validate and sanitize activation tensor for numerical stability."""
        try:
            if activation is None:
                print(f"Warning: Received None activation{f' in {context}' if context else ''}")
                return None
                
            # Handle potential inf/nan values
            if torch.isnan(activation).any() or torch.isinf(activation).any():
                # Replace inf/nan with safe values
                activation = torch.nan_to_num(
                    activation,
                    nan=0.0,
                    posinf=torch.finfo(torch.float32).max,
                    neginf=torch.finfo(torch.float32).min
                )
            
            # Clip extremely large values to float32 range
            activation = torch.clamp(
                activation,
                min=torch.finfo(torch.float32).min,
                max=torch.finfo(torch.float32).max
            )
            
            # Safely convert to float32
            activation = activation.to(torch.float32)
            
            return activation
            
        except Exception as e:
            print(f"Error validating activation{f' in {context}' if context else ''}: {str(e)}")
            return None


    def _qwen2_process_activation(self, activation: torch.Tensor) -> Optional[torch.Tensor]:
        """Process Qwen2 model attention output activations"""
        try:
            # Validate and sanitize activation
            activation = self._validate_and_sanitize_activation(activation, "qwen2_process_activation")
            if activation is None:
                return None
                
            # For Qwen2 attention output, shape is [batch_size, seq_len, hidden_size]
            if len(activation.shape) == 3:
                batch_size, seq_len, hidden_dim = activation.shape
                
                # Verify the dimension matches hidden size
                if hidden_dim != self.hidden_size:
                    print(f"Warning: Unexpected hidden dimension. Expected {self.hidden_size}, got {hidden_dim}")
                    return None
                    
                # Reshape into attention head format, accounting for grouped-query attention
                # Qwen2 uses different number of heads for keys/values vs queries
                if self.num_heads > self.kv_heads:
                    # Handle grouped-query attention case
                    # Each key-value head is shared across num_heads/kv_heads query heads
                    head_ratio = self.num_heads // self.kv_heads
                    processed = activation.view(batch_size, seq_len, self.num_heads, -1)
                    # Repeat the values for each query head in the group
                    processed = processed.repeat_interleave(head_ratio, dim=2)
                else:
                    # Standard case where num_heads equals kv_heads
                    processed = activation.view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                return processed.detach().cpu()
            
            return None
            
        except Exception as e:
            print(f"Error processing Qwen2 activation: {str(e)}")
            return None
        
    def temp_test(self, query: str, task_name: str) -> TaskActivations:
        self.attention_activations.clear()

        try:
            model_inputs = self.model_tokenizer(
                query,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            attention_activations = self._process_head_features()

            return TaskActivations(task_name=task_name, attention_activations=attention_activations)
        
        except Exception as e:
            print(f"Error computing activations: {str(e)}")
            raise

        
