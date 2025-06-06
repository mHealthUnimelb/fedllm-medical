import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from benchmarks import infer_medmcqa, infer_medqa, infer_pubmedqa, infer_careqa

# Fixed seed
torch.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base-model-name-path", type=str, default="Qwen/Qwen2.5-7B-Instruct"
)
parser.add_argument("--run-name", type=str, default="fl")
parser.add_argument("--peft-path", type=str, default=None)
parser.add_argument(
    "--datasets",
    type=str,
    default="pubmedqa",
    help="The dataset to infer on: [pubmedqa, medqa, medmcqa]",
)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--quantization", type=int, default=4)
args = parser.parse_args()

print(args.peft_path)

# Load model and tokenizer
if args.quantization == 4:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    torch_dtype = torch.float32
elif args.quantization == 8:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    torch_dtype = torch.float16
else:
    raise ValueError(
        f"Use 4-bit or 8-bit quantization. You passed: {args.quantization}/"
    )

model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name_path,
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
if args.peft_path is not None:
    model = PeftModel.from_pretrained(
        model, args.peft_path, torch_dtype=torch_dtype
    ).to("cuda")

for name, param in model.named_parameters():
    # 3. Check if the parameter is a LoRA parameter
    if "lora_" in name:
        print(f"Parameter Name: {name}")
        print(f"Parameter Shape: {param.shape}")
        print(f"Parameter Device: {param.device}")
        print(f"Parameter Mean: {param.data.mean()}")
        print(f"Parameter Std Dev: {param.data.std()}")
        print("-" * 40)


tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_path)

# Evaluate
for dataset in args.datasets.split(","):
    if dataset == "pubmedqa":
        infer_pubmedqa(model, tokenizer, args.batch_size, args.run_name)
    elif dataset == "medqa":
        infer_medqa(model, tokenizer, args.batch_size, args.run_name)
    elif dataset == "medmcqa":
        infer_medmcqa(model, tokenizer, args.batch_size, args.run_name)
    elif dataset == "careqa":
        infer_careqa(model, tokenizer, args.batch_size, args.run_name)
    else:
        raise ValueError("Undefined Dataset.")
