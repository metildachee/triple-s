# Triple-S: Sticker Semantic Similarity Benchmark and GSE  

This repository contains the resources for the paper **"Sticker Semantic Similarity Benchmark and General Sticker Encoder (GSE)"**.  

We are uploading the dataset and weights....

---

## Repository Structure  

- `weights/`  
  Contains the trained weights for our General Sticker Encoder (GSE).  

- `dataset/`  
  Includes the benchmark data:  
  - `train.csv` – training set  
  - `test.csv` – evaluation set  

---

## Base Models  

- [CLIP (ViT-B/32)](https://huggingface.co/openai/clip-vit-base-patch32)  
- Images sourced from [Sticker-Queries Dataset](https://huggingface.co/datasets/metchee/sticker-queries)  

---

## Usage  

### Generate Embeddings  

Make the script executable and run:  

```bash
chmod +x generate_embeddings.sh
./generate_embeddings.sh
