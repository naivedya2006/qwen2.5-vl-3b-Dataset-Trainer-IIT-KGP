# 🚦 Qwen2.5-VL Fine-Tuning Framework for Pedestrian Crossing Severity (IIT-KGP)

A production-ready, highly optimized Python framework for fine-tuning **Qwen2.5-VL (3B and 7B)** Vision-Language Models. This repository is specifically made for multimodal severity prediction of pedestrian crossings, utilizing LoRA/QLoRA for consumer-hardware compatibility (e.g., RTX 4070 12GB on Windows).

BuILT with Hugging Face `transformers`, `peft`, `accelerate`, and `bitsandbytes`.

---

## 🌟 Key Features

- **Plug-and-Play Dataset Scaling**: Automatically parses `chatml_train.json` and loads matching media. Supports dynamic scaling from 50 to 10,000+ pedestrian records without any code modification.
- **Consumer Hardware Optimized**: Pre-configured for an RTX 4070 (12GB VRAM) using 4-bit QLoRA, Gradient Checkpointing, Flash Attention 2 / SDPA, and micro-batching.
- **Robust Training Pipeline**: Features automatic checkpoint discovery and resumption, cosine learning rate scheduling, mixed precision (`bfloat16`), and TensorBoard logging.
- **Modular Architecture**: Clean, object-oriented separation of configurations, dataset loading, data collation, model preparation, and inference.

---

## 📂 Folder Structure

The framework expects your pre-existing `Qwen2.5VL_Dataset` to be placed adjacent to or specified via path to the `qwen_finetune` directory.

```text
📦 Project_Root
 ┣ 📂 Qwen2.5VL_Dataset/           # YOUR EXISTING DATASET (Unmodified)
 ┃ ┣ 📂 clips/                     # Pedestrian video clips (.mp4)
 ┃ ┣ 📂 context_frames/            # Pedestrian context images (.jpg)
 ┃ ┣ 📂 metadata/                  # JSON metadata per pedestrian
 ┃ ┣ 📜 chatml_train.json          # Training conversation formats
 ┃ ┣ 📜 chatml_val.json            # Validation conversation formats
 ┃ ┗ 📜 chatml_test.json           # Testing conversation formats
 ┃
 ┗ 📂 qwen_finetune/               # THIS FRAMEWORK
   ┣ 📂 logs/                      # Training logs (Auto-generated)
   ┣ 📂 runs/                      # TensorBoard & Checkpoints (Auto-generated)
   ┣ 📜 config.py                  # Hyperparameters and model configuration
   ┣ 📜 dataset.py                 # Lazy-loading multimodal PyTorch Dataset
   ┣ 📜 collator.py                # Padding and batch preparation
   ┣ 📜 model_loader.py            # Qwen2.5-VL initialization and QLoRA prep
   ┣ 📜 lora.py                    # PEFT/LoRA adapter application
   ┣ 📜 trainer.py                 # Hugging Face Trainer configuration
   ┣ 📜 metrics.py                 # Evaluation metrics and plots
   ┣ 📜 train.py                   # Main training entry point
   ┣ 📜 inference.py               # Programmatic inference engine
   ┣ 📜 predict.py                 # CLI wrapper for inference
   ┣ 📜 requirements.txt           # Python dependencies
   ┗ 📜 README.md                  # This documentation
```

---

## ⚙️ Installation & Requirements

Ensure you are running **Python 3.10 or 3.11** on Windows/Linux with CUDA 12.x installed.

1. **Clone/Navigate to the directory**:
   ```bash
   cd qwen_finetune
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note for Windows users: `bitsandbytes` >= 0.41.1 supports Windows natively. Ensure your PyTorch installation matches your CUDA version.*

3. **(Optional but Recommended) Install Flash Attention 2**:
   If you have a compatible build environment:
   ```bash
   pip install flash-attn --no-build-isolation
   ```
   *If installation fails on Windows, the framework will gracefully fall back to PyTorch's native Scaled Dot Product Attention (SDPA).*

---

##  Tutorial

### Step 1: Verify Configuration (`config.py`)
Open `config.py` to verify or adjust hyperparameters. The defaults are strictly tuned to prevent Out-Of-Memory (OOM) errors on a 12GB GPU:
- `use_qlora = True`: Uses 4-bit quantization.
- `per_device_train_batch_size = 1`: Keeps memory footprint low.
- `gradient_accumulation_steps = 8`: Simulates a batch size of 8.
- `max_video_frames = 8`: Limits frames processed per video.
- `dataloader_num_workers = 0`: **Required on Windows** to prevent multi-processing spawn errors.

*To switch to the 7B model, simply change:*
```python
model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
```

### Step 2: Start Training
Run the main training script. The framework will automatically validate the dataset structure, apply LoRA, and begin training.
```bash
python train.py
```
**Resuming Training**: If your run is interrupted, simply run `python train.py` again. The framework detects the `runs/` directory and resumes from the latest checkpoint.

### Step 3: Monitor Progress
Open a new terminal and launch TensorBoard to monitor loss and metrics in real-time:
```bash
tensorboard --logdir runs/
```

### Step 4: Run Inference (Do not use inference script as they are still in production state thanks:)
Once training is complete (or using an intermediate checkpoint), you can test the model on a specific pedestrian clip using the CLI tool:

```bash
python predict.py \
  --video ../Qwen2.5VL_Dataset/clips/Ped_1.mp4 \
  --image ../Qwen2.5VL_Dataset/context_frames/Ped_1.jpg \
  --instruction "Analyze the pedestrian crossing severity." \
  --lora_path runs/qwen2.5-vl-finetune/best_model
```
*Output will stream the model's reasoning, behavior explanation, and final severity prediction directly to the console.*

---

##  Under the Hood

- **Dataset Handling (`dataset.py`)**: Uses a lazy-loading strategy. Only the textual ChatML is held in RAM. Videos and images are resolved and processed via the `qwen_vl_utils.process_vision_info` pipeline precisely at fetch time.
- **Collator (`collator.py`)**: Dynamically pads textual tokens and constructs the 3D visual tensors. Ensures `-100` is applied to pad tokens so they are ignored during loss calculation.
- **Memory Efficiency**: Limits visual resolution to `100,000` pixels per media input to guarantee execution under 12GB VRAM constraints while preserving aspect ratios.

---

## ⚠️ Troubleshooting & Best Practices

1. **CUDA Out of Memory (OOM)**:
   - Decrease `max_video_frames` in `config.py` from `8` to `4`.
   - Ensure `use_qlora = True` is active.
   - Close any other applications using GPU VRAM (e.g., browsers, other Python scripts).

2. **Windows Multiprocessing Error (BrokenPipe / EOF)**:
   - Ensure `dataloader_num_workers` in `config.py` remains set to `0`. PyTorch on Windows handles multiprocessing poorly in Jupyter/standard scripts without a `__main__` guard.

3. **Model Downloading Too Slowly**:
   - The initial run will download the 3B/7B model from Hugging Face (~6GB to ~15GB). Set the environment variable `HF_HUB_ENABLE_HF_TRANSFER=1` and `pip install hf_transfer` for faster downloads.

4. **Missing Dataset Files**:
   - The framework strictly expects `chatml_train.json`. Ensure your relative path in `config.py` (`dataset_dir = "../Qwen2.5VL_Dataset"`) correctly points to the dataset root.
