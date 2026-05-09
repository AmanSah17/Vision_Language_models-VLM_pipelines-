# Vision-Language Model (VLM) PyTorch Fundamentals

## Overview
This notebook demonstrates the step-by-step construction, training, and evaluation of a simple Vision-Language Model (VLM) using PyTorch and HuggingFace Transformers. The model combines a Vision Transformer (ViT) as an image encoder and a Llama-based language model as a text decoder, enabling image captioning tasks.

### Key Steps
1. **Dataset Preparation**: Download and preprocess a dataset of images and captions.
2. **Reproducibility**: Set seeds for deterministic results.
3. **Configuration**: Define all hyperparameters and model settings.
4. **Tokenizer Setup**: Load and configure the Llama tokenizer.
5. **Image Preprocessing**: Convert images to base64 and prepare DataFrame.
6. **Vision Transformer (ViT) Setup**: Configure and test the image encoder.
7. **Model Architecture**: Build a custom VLM class combining ViT and Llama.
8. **Image Captioning Demo**: Run a forward pass and generate captions for a sample image.
9. **Dataset Class**: Implement a PyTorch Dataset for training.
10. **Training Loop**: Train the model and evaluate on a validation set.
11. **Inference**: Generate captions for new images.

---

## Implementation Plan

### 1. Data Acquisition & Preprocessing
- Clone a HuggingFace dataset containing images and captions.
- Convert images to base64 for efficient storage in DataFrames.
- Prepare a DataFrame with image and caption columns.

### 2. Reproducibility
- Set random seeds for Python, NumPy, and PyTorch.
- Configure PyTorch for deterministic behavior.

### 3. Configuration
- Define all model and training hyperparameters (batch size, learning rate, model dimensions, etc.).

### 4. Tokenizer
- Load the Llama tokenizer and set padding/eos tokens.

### 5. Image Preprocessing
- Define functions to convert image files to base64 and back to tensors.
- Normalize images using ImageNet statistics.

### 6. Vision Transformer (ViT)
- Configure and instantiate a ViT model for image encoding.
- Test the output shape for a dummy input.

### 7. Vision-Language Model (VLM)
- Build a custom PyTorch module combining ViT and Llama.
- Project image embeddings to match Llama's embedding size.
- Concatenate image and text embeddings for joint processing.
- Implement forward and generation methods.

### 8. Dataset & DataLoader
- Implement a custom Dataset class to load (image, caption) pairs.
- Tokenize captions and prepare input/target tensors.
- Split data into training and validation sets.

### 9. Training & Evaluation
- Define training and validation loops.
- Track and print loss during training.

### 10. Inference
- Generate captions for new images using the trained model.

---

## Mathematical Formulation

Given an image $I$ and its caption $C = (w_1, w_2, ..., w_n)$, the model learns $P(C|I)$, i.e., the probability of the caption given the image. The loss is the cross-entropy between the predicted and true tokens:

$$
\mathcal{L} = -\sum_{t=1}^n \log P(w_t | I, w_{<t})
$$

The ViT encodes $I$ to a vector $v_I$, which is projected and concatenated with the token embeddings of $C$ before being fed to the Llama decoder.

---

## Parameters & Configuration
- **BATCH_SIZE**: Number of samples per batch.
- **N_HIDDEN_LAYERS**: Number of transformer layers in ViT.
- **MAX_LENGTH**: Maximum caption length.
- **EVAL_INTERVAL**: Steps between validation checks.
- **LEARNING_RATE**: Adam optimizer learning rate.
- **EPOCHS**: Number of training epochs.
- **N_EMBD**: Embedding dimension for Llama.
- **N_HEAD**: Number of attention heads.
- **N_LAYER**: Number of Llama layers.
- **DROPOUT**: Dropout probability.
- **IMG_SIZE**: Image size (pixels).
- **PATCH_SIZE**: Patch size for ViT.
- **IMAGE_EMBED_DIM**: Output dimension of ViT.
- **N_CHANNELS**: Number of image channels (RGB=3).
- **MAX_POSITION_EMBEDDINGS**: Max sequence length for Llama.

---

## Usage
- Each code cell is preceded by a markdown cell explaining its purpose, technical details, and rationale.
- Mathematical and technical explanations are provided where relevant.
- The notebook is structured for clarity, reproducibility, and educational value.

---

## References
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [Llama Language Model](https://huggingface.co/docs/transformers/model_doc/llama)
