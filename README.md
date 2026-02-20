# SDXL GGUF Quantize Tool

A complete, standalone local GUI tool for extracting SDXL checkpoint components and quantizing the UNet to GGUF format. This tool completely eliminates the need for complex Python setups, Jupyter Notebooks, or manual terminal commands by wrapping everything in an easy-to-use cross-platform interface.

## 🌟 Key Features

- **True 1-Click Portable Execution:** Works seamlessly on both Windows and Linux/Colab. No complex environment setup required.
- **Graphical User Interface (GUI):** A clean, dark-mode CustomTkinter interface for local users.
- **Web UI (Gradio):** A Gradio wrapper optimized for headless environments like Google Colab.
- **Smart Component Extraction:** Extracts the UNet and optionally bundles CLIP models with their original prefixes preserved (crucial for compatible loading in ComfyUI custom nodes).
- **Asynchronous Execution:** Highly responsive UI that will not freeze during intensive 20GB+ processing loads, complete with real-time logs and a safe Stop button.

## 🚀 Installation & Standard Usage

The easiest way to use this tool is to use the **Portable Release**.

1. Download the latest `SDXL_GGUF_Quantize_Tool_Portable.zip` from [**GitHub Releases**](#).
2. Extract the folder to your desired location.
3. **Windows Users:** Double-click `run.bat`
4. **Linux / Colab Users:** Run `bash run.sh`

The launcher will dynamically create a local virtual environment, discreetly install required dependencies, and automatically load the application.

### ☁️ Cloud Usage (Google Colab)

If you prefer to run the extraction and quantization pipeline entirely in the cloud, we provide a ready-to-use Google Colab notebook!

👉 [Open **SDXL_Quantize_Tool.ipynb** in Google Colab](https://colab.research.google.com/github/magekinnarus/SDXL_GGUF_Quantize_Tool/blob/main/SDXL_Quantize_Tool.ipynb) *(Update link if repository name is different)*

This notebook utilizes the same powerful unified backend but wraps it in a cloud-friendly Gradio web interface.

## 🛠️ Usage Breakdown

The application is divided into four main pipeline tabs depending on your needs:

#### 1. Full Pipeline
Extracts both the UNet and the CLIP models from a standard SDXL `.safetensors` checkpoint, converts the UNet to `F16.gguf` format, and finally quantizes it into your chosen GGUF formats (like `Q8_0`, `Q5_K_M`, etc.).

#### 2. Model Extraction Only
Only runs the extraction stage. Splits a standard `.safetensors` checkpoint into two separate `.safetensors` files: one containing just the UNet, and one containing just the CLIPs.

#### 3. Quantize Only
Skips extraction. Feed it an already extracted UNet `.safetensors` file (or an existing `F16.gguf` file) and generate smaller, highly-efficient quantized GGUF variants.

#### 4. Batch Process
Define an input folder full of `.safetensors` checkpoints and an output folder. The tool will cleanly iterate through every model, extracting and quantizing them consecutively.

***

## 🧠 Technical Details

### ⚠️ CRITICAL: Custom ComfyUI Node Required for CLIPs

If you used the legacy 2024 SDXL extraction notebook, you likely remember that the CLIPs patched through a number of processesso they could load in ComfyUI's standard `DualClipLoader`. **This tool no longer does that.** 

We now extract "Bundled CLIPs" (with their original prefixes preserved). This means **they will fail to load** in standard ComfyUI CLIP loaders. 

To use the CLIPs extracted by this tool in ComfyUI, you **must** use the custom `DJ_Cliploader` node:
👉 [Download ComfyUI-DJ_nodes Here](https://github.com/magekinnarus/ComfyUI-DJ_nodes)

This custom node handles the dynamic prefix resolution at load-time securely, rather than applying destructive stripping to the files during quantization.

### Cross-Platform C++ Backend
The underlying quantization logic relies on modified forks of `llama.cpp`. The portable release contains pre-compiled `.dll` (Windows) and `.so` (Linux) shared libraries alongside the `llama-quantize` binaries. The python wrapper seamlessly interfaces with the correct system architectures dynamically without ever exposing C++ compilation to the end user.
