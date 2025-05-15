# -*- coding: utf-8 -*-

!pip install datasets huggingface_hub

!pip install datasets
from datasets import load_dataset

from huggingface_hub import login

# Paste your HF token here
from datasets import load_dataset

# Load the dataset
dataset = load_dataset(
    "BAAI/OPI",
    data_files="OPI_DATA/AP/Function/test/UniProtSeq_function_test.jsonl",
    split="train"
)

print(dataset[0])

print(dataset[500])





# Step 1: Install dependencies
!pip install -q transformers datasets huggingface_hub

# Step 2: Imports
import json
import os
import requests
import torch
from transformers import AutoTokenizer, EsmModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Step 3: Load ESM-3 model (esm2 is used here as ESM-3 is not yet public on HF)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval().cuda()

# Step 4: Download CASPSimilarSeq_function_test.jsonl manually
url = "https://huggingface.co/datasets/BAAI/OPI/resolve/main/OPI_DATA/AP/Function/test/CASPSimilarSeq_function_test.jsonl"
filename = "CASPSimilarSeq_function_test.jsonl"

if not os.path.exists(filename):
    response = requests.get(url)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(" Downloaded CASPSimilarSeq_function_test.jsonl")

# Step 5: Parse JSONL and extract examples
examples = []
with open(filename, "r") as f:
    for line in f:
        data = json.loads(line)
        for inst in data["instances"]:
            examples.append({
                "input": inst["input"],
                "output": inst["output"]
            })

print(f" Loaded {len(examples)} examples")

# Step 6: Embed sequences with ESM
def get_esm_embedding(sequence):
    tokens = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        outputs = model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

embeddings, labels = [], []

for i, ex in enumerate(examples[:50]):  # keep small for Colab
    try:
        emb = get_esm_embedding(ex["input"])
        embeddings.append(emb)
        labels.append(ex["output"])
        print(f"✅ Embedded {i+1}: {ex['output'][:40]}...")
    except Exception as e:
        print(f"⚠️ Error at {i}: {e}")

# Step 7: Compute similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Step 8: Plot
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap="viridis")
plt.title("ESM Functional Similarity - CASPSimilarSeq")
plt.colorbar(label="Cosine Similarity")
plt.xlabel("Protein Index")
plt.ylabel("Protein Index")
plt.show()

!pip install esm datasets

import json
import requests

# URL to the dataset file
dataset_url = "https://huggingface.co/datasets/BAAI/OPI/resolve/main/OPI_DATA/AP/Function/test/CASPSimilarSeq_function_test.jsonl"

# Download and parse the JSONL file
response = requests.get(dataset_url)
data = [json.loads(line) for line in response.text.splitlines()]

# Display the first entry to understand its structure
print(data[0])

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/evolutionaryscale/esm.git
# %cd esm
!pip install -e .

!pip install --quiet transformers datasets accelerate
!pip install --quiet git+https://github.com/facebookresearch/esm.git
from huggingface_hub import notebook_login

# Only needed once per session
notebook_login()

from datasets import load_dataset

dataset = load_dataset(
    "BAAI/OPI",
    data_files={"test": "OPI_DATA/AP/Function/test/CASPSimilarSeq_function_test.jsonl"},
    split="test",
    streaming=False
)

print(dataset[0])  # Sanity check

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

import torch
import numpy as np

def embed_sequence(seq):
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    token_representations = outputs.last_hidden_state
    return token_representations[0, 1:len(seq)+1].mean(dim=0).numpy()

embeddings = []
labels = []

for i in range(50):  # Use 50 examples to avoid OOM
    seq = dataset[i]["instances"][0]["input"]
    label = dataset[i]["instances"][0]["output"]
    emb = embed_sequence(seq)
    embeddings.append(emb)
    labels.append(label)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = pd.DataFrame(embeddings)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

