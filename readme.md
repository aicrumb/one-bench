How to run One-Bench, the benchmark to end all benchmarks.

```python
import transformers
import torch

model_id = "facebook/opt-6.7b" # substitute with your huggingface repo id
device = 0

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = {"": device},
    torch_dtype = torch.bfloat16,
    _attn_implementation = "flash_attention_2",
    #optional: quantization_config = transformers.BitsAndBytesConfig(
    #        load_in_4bit=True,
    #        bnb_4bit_compute_dtype=torch.bfloat16,
    #        bnb_4bit_use_double_quant=True,
    #        bnb_4bit_quant_type="nf4",
    #    )
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
onebench = load_dataset("crumb/onebench", split="train")

def ln_ppl_inefficient(x):
    with torch.no_grad():
        ids = tokenizer(x['text'], return_tensors='pt').input_ids.cuda()
        return (2.718281828459045 ** (model(
            ids,
            labels=ids,
        ).loss.item())) / ids.size(1)

onebench = onebench.map(
    lambda x: {
        'ppl': ln_ppl_inefficient(x)
    },
    num_proc = 1,
    batched = False
)

print(sum([i['ppl'] for i in data]) / len(data))
```
