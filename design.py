
!pip install transformers
!pip install accelerate

pip install esm

 # Replace with your token 

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import json
from tqdm import tqdm

# Step 1: Login to Hugging Face
login()  # You'll be prompted to paste your HF token (read access is enough)

# Step 2: Load the model from Hugging Face
model = ESM3.from_pretrained("esm3-open").to("cuda")  # or "cpu"

pip install esm

# Load OPI data
with open("/content/drive/MyDrive/OPI_train.json", "r") as f:
    data = json.load(f)

# Filter for functional keyword task
task = "functional keyword"
keyword_data = [x for x in data if task in x["instruction"].lower()]

with open("/content/drive/MyDrive/OPI_es3_augmented_keywords.json", "w") as f:
    json.dump(augmented, f, indent=2)

print(" Saved ESM3-enhanced keyword data.")

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import json
from tqdm import tqdm
import re

login()
model = ESM3.from_pretrained("esm3-open").to("cuda")

with open("/content/drive/MyDrive/OPI_train.json", "r") as f:
    raw_data = json.load(f)

def clean_sequence(seq):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)

def safe_generate(protein, gen_config):
    from esm.sdk.api import ESMProteinError
    try:
        result = model.generate(protein, gen_config)
        if isinstance(result, ESMProteinError):
            raise ValueError("Protein generation failed internally.")
        return result
    except Exception as e:
        raise ValueError(f"ESM3 error: {e}")

results = []
fail_count = 0
success_count = 0

for item in tqdm(raw_data[:100]):
    try:
        original_seq = item["input"]
        clean_seq = clean_sequence(original_seq)
        if len(clean_seq) < 50 or len(clean_seq) > 1024:
            continue
        if len(set(clean_seq)) < 5:
            continue
        protein = ESMProtein(sequence=clean_seq)
        protein = safe_generate(protein, GenerationConfig(track="structure", num_steps=8))
        structure1 = protein.coordinates
        if structure1 is None:
            raise ValueError("No structure generated in step 1.")
        protein.sequence = None
        protein = safe_generate(protein, GenerationConfig(track="sequence", num_steps=8))
        new_seq = protein.sequence
        if new_seq is None or len(new_seq) < 50:
            raise ValueError("Sequence regeneration failed.")
        protein.coordinates = None
        protein = safe_generate(protein, GenerationConfig(track="structure", num_steps=8))
        structure2 = protein.coordinates
        if structure2 is None:
            raise ValueError("No structure generated in round-trip.")
        results.append({
            "instruction": item["instruction"],
            "original_input": original_seq,
            "clean_input": clean_seq,
            "regenerated_sequence": new_seq,
            "structure_start": structure1.tolist(),
            "structure_roundtrip": structure2.tolist(),
            "output": item["output"]
        })
        success_count += 1
    except Exception as e:
        fail_count += 1
        print(f" Failed on item: {item['input'][:30]}... — {e}")
        continue

with open("/content/drive/MyDrive/OPI_inverse_design_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Done. Success: {success_count}, Failures: {fail_count}, Total Processed: {success_count + fail_count}")
