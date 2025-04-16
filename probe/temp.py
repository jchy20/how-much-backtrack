from activation import ActivationManager, TaskActivations
import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model.to(device)


action_manager = ActivationManager(model, tokenizer, device)
result = action_manager.temp_test("What is the capital of France?", "QA")

with open('data.pkl', 'wb') as f:
    pickle.dump(result.attention_activations, f)