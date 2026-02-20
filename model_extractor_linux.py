import os
import argparse
import subprocess
import torch
import json
import time
import threading
from safetensors.torch import load_file, save_file


class PipelineCancelled(Exception):
    pass


def _quant_output_path(output_dir, model_name, fmt):
    suffix = "Q8" if fmt == "Q8_0" else fmt
    return os.path.join(output_dir, "quantized", f"{model_name}_{suffix}.gguf")


def _cleanup_incomplete_paths(paths, preexisting_paths=None, log_callback=None):
    preexisting_paths = preexisting_paths or set()
    for path in paths:
        if not path or path in preexisting_paths:
            continue
        if os.path.isfile(path):
            try:
                os.remove(path)
                _emit(f"Cleanup removed incomplete file: {path}", log_callback)
            except OSError as exc:
                _emit(f"Cleanup warning for {path}: {exc}", log_callback)

def _emit(message, log_callback=None):
    if log_callback:
        log_callback(message)
    else:
        print(message)

def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("Process interrupted by user.")


def run_command(command, cwd=None, shell=True, log_callback=None, cancel_event=None):
    _check_cancel(cancel_event)
    _emit(f"Running: {command}", log_callback)
    child_env = os.environ.copy()
    # Make Python child processes stream logs immediately into the GUI.
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    output_lock = threading.Lock()

    def _read_output():
        if process.stdout is None:
            return
        for line in process.stdout:
            # Handle tqdm and carriage returns
            if '\r' in line:
                line = line.split('\r')[-1]
            line = line.rstrip()
            if line:
                with output_lock:
                    _emit(line, log_callback)

    reader = threading.Thread(target=_read_output, daemon=True)
    reader.start()

    while True:
        return_code = process.poll()
        if return_code is not None:
            break
        if cancel_event is not None and cancel_event.is_set():
            _emit("Interrupt requested. Stopping current subprocess...", log_callback)
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            reader.join(timeout=1)
            raise PipelineCancelled("Process interrupted by user.")
        time.sleep(0.1)

    reader.join(timeout=1)
    if return_code != 0:
        _emit(f"Error running command: {command}", log_callback)
        raise subprocess.CalledProcessError(return_code, command)

def validate_tools(tools_dir):
    missing = []
    convert_script = os.path.join(tools_dir, "convert.py")
    quantize_bin = os.path.join(tools_dir, "bin", "llama-quantize")
    
    # Ensure binary is executable on Linux
    if os.path.isfile(quantize_bin):
        import stat
        st = os.stat(quantize_bin)
        os.chmod(quantize_bin, st.st_mode | stat.S_IEXEC)
        
    if not os.path.isfile(convert_script):
        missing.append(convert_script)
    if not os.path.isfile(quantize_bin):
        missing.append(quantize_bin)
    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_lines}")

def extract_components(
    model_path,
    output_dir,
    model_name,
    overwrite=False,
    log_callback=None,
    step_progress_callback=None,
    cancel_event=None,
    skip_unet=False,
    skip_clips=False,
):
    _check_cancel(cancel_event)
    
    output_dir = os.path.abspath(output_dir)
    unet_output = os.path.join(output_dir, 'unet', f'{model_name}_unet.safetensors')
    clips_output = os.path.join(output_dir, 'clips', f'{model_name}_clips.safetensors')

    # If both skipped, nothing to do
    if skip_unet and skip_clips:
        return unet_output

    _emit(f"Initializing extraction for: {os.path.basename(model_path)}", log_callback)
    _emit("Reading safetensors (this may take a moment)...", log_callback)
    time.sleep(0.2) # Yield to let UI render these logs before heavy load
    start_ts = time.perf_counter()
    state_dict = load_file(model_path)
    total_tensors = max(len(state_dict), 1)
    _emit(f"Loaded state_dict with {total_tensors} tensors.", log_callback)
    time.sleep(0.1)
    if step_progress_callback:
        step_progress_callback(0.05, "Loaded model tensors")
    
    os.makedirs(os.path.dirname(unet_output), exist_ok=True)
    os.makedirs(os.path.dirname(clips_output), exist_ok=True)

    # Overwrite check for extracted files
    if not skip_unet and os.path.exists(unet_output) and overwrite:
        _emit(f"Overwriting existing file: {unet_output}", log_callback)
        os.remove(unet_output)
    if not skip_clips and os.path.exists(clips_output) and overwrite:
        _emit(f"Overwriting existing file: {clips_output}", log_callback)
        os.remove(clips_output)

    # 1. Extract UNet
    if not skip_unet:
        # Check if exists and not overwrite
        if os.path.exists(unet_output) and not overwrite:
             _emit(f"UNet already exists, skipping extraction: {unet_output}", log_callback)
        else:
            _emit("Extracting UNet...", log_callback)
            unet_dict = {}
            unet_update_interval = max(total_tensors // 20, 1) # Reduce updates to 20 total
            last_unet_log_ts = time.perf_counter()
            for idx, (k, v) in enumerate(state_dict.items(), start=1):
                _check_cancel(cancel_event)
                if k.startswith("model_ema"):
                     continue
                if not k.startswith("conditioner.") and not k.startswith("first_stage_model."):
                    key = k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"
                    unet_dict[key] = v
                
                if step_progress_callback and (idx % unet_update_interval == 0 or idx == total_tensors):
                    step_progress_callback(0.05 + 0.70 * (idx / total_tensors), "Extracting UNet tensors")
                # Yield to let UI process events
                if idx % 100 == 0:
                    time.sleep(0.005)
            
            _emit(f"Saving UNet to {unet_output}...", log_callback)
            time.sleep(0.1)
            save_file(unet_dict, unet_output)
            _emit(f"Saved UNet successfully.", log_callback)
            del unet_dict

    # 2. Extract and Combine CLIPs
    if not skip_clips:
        if os.path.exists(clips_output) and not overwrite:
             _emit(f"CLIPs already exists, skipping extraction: {clips_output}", log_callback)
        else:
            _emit("Extracting and Combining CLIPs...", log_callback)
            clips_dict = {}
            clips_update_interval = max(total_tensors // 10, 1) # Reduce updates to 10 total
            last_clips_log_ts = time.perf_counter()
            for idx, (k, v) in enumerate(state_dict.items(), start=1):
                _check_cancel(cancel_event)
                if k.startswith("conditioner.embedders."):
                    clips_dict[k] = v
                
                if step_progress_callback and (idx % clips_update_interval == 0 or idx == total_tensors):
                    step_progress_callback(0.75 + 0.25 * (idx / total_tensors), "Extracting CLIP tensors")
                # Yield to let UI process events
                if idx % 100 == 0:
                    time.sleep(0.005)
            _emit(f"Saving Combined CLIPs to {clips_output}...", log_callback)
            time.sleep(0.1)
            save_file(clips_dict, clips_output)
            _emit(f"Saved Combined CLIPs successfully.", log_callback)
            
            del clips_dict
    
    del state_dict
    elapsed = time.perf_counter() - start_ts
    _emit(f"Extraction stage completed in {elapsed:.1f}s", log_callback)
    if step_progress_callback:
        step_progress_callback(1.0, "Extraction step complete")
    return unet_output

def convert_to_gguf(
    unet_path,
    output_dir,
    model_name,
    tools_dir,
    overwrite=False,
    log_callback=None,
    cancel_event=None,
):
    _check_cancel(cancel_event)
    output_dir = os.path.abspath(output_dir)
    f16_output_dir = os.path.join(output_dir, 'fp16')
    os.makedirs(f16_output_dir, exist_ok=True)
    f16_output = os.path.join(f16_output_dir, f'{model_name}_F16.gguf')
    
    if os.path.exists(f16_output):
        if overwrite:
            _emit(f"Overwriting existing FP16: {f16_output}", log_callback)
            os.remove(f16_output)
        else:
            _emit(f"FP16 GGUF already exists, skipping conversion: {f16_output}", log_callback)
            return f16_output

    convert_script = os.path.join(tools_dir, 'convert.py')
    # Use --src and --dst only
    cmd = f'python -u "{convert_script}" --src "{unet_path}" --dst "{f16_output}"'
    run_command(cmd, log_callback=log_callback, cancel_event=cancel_event)
    
    return f16_output

def quantize_model(
    f16_path,
    output_dir,
    model_name,
    formats,
    tools_dir,
    overwrite=False,
    log_callback=None,
    cancel_event=None,
):
    _check_cancel(cancel_event)
    output_dir = os.path.abspath(output_dir)
    quantized_output_dir = os.path.join(output_dir, 'quantized')
    os.makedirs(quantized_output_dir, exist_ok=True)
    
    # Path to your confirmed Linux executable
    quantize_bin = os.path.join(tools_dir, 'bin', 'llama-quantize')

    for fmt in formats:
        _check_cancel(cancel_event)
        out_file = _quant_output_path(output_dir, model_name, fmt)
        
        if os.path.exists(out_file) and overwrite:
            _emit(f"Overwriting existing quantized model: {out_file}", log_callback)
            os.remove(out_file)
            
        cmd = f'"{quantize_bin}" "{f16_path}" "{out_file}" {fmt}'
        run_command(cmd, log_callback=log_callback, cancel_event=cancel_event)

def process_model(
    input_path,
    output_dir,
    formats=None,
    overwrite=False,
    tools_dir=None,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
    cleanup_incomplete=True,
    skip_unet=False,
    skip_clips=False,
):
    _check_cancel(cancel_event)
    if tools_dir is None:
        tools_dir = os.path.dirname(os.path.abspath(__file__))
    if formats is None:
        formats = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    formats = [fmt.strip() for fmt in formats if fmt and fmt.strip()]
    if not formats:
        raise ValueError("No quantization formats selected.")

    validate_tools(tools_dir)

    model_name = os.path.splitext(os.path.basename(input_path))[0]
    # Strip common extraction suffixes (case-insensitive)
    for suffix in ["_unet", ".unet", "_f16", ".f16", "-f16"]:
        if model_name.lower().endswith(suffix):
            model_name = model_name[:-len(suffix)]
            
    output_dir = os.path.abspath(output_dir)
    unet_output = os.path.join(output_dir, "unet", f"{model_name}_unet.safetensors")
    clips_output = os.path.join(output_dir, "clips", f"{model_name}_clips.safetensors")
    f16_output = os.path.join(output_dir, "fp16", f"{model_name}_F16.gguf")
    quant_outputs = {fmt: _quant_output_path(output_dir, model_name, fmt) for fmt in formats}
    
    tracked_paths = []
    if not skip_unet:
        tracked_paths.append(unet_output)
    if not skip_clips:
        tracked_paths.append(clips_output)
    tracked_paths.append(f16_output)
    tracked_paths.extend(list(quant_outputs.values()))
    
    preexisting_paths = {path for path in tracked_paths if os.path.exists(path)}
    completed_paths = set()

    total_steps = 2 + len(formats)
    if skip_unet and skip_clips:
        total_steps = len(formats) # Only quantizing
        if input_path.lower().endswith(".gguf"):
             f16_output = input_path # Input is already GGUF
        else:
             total_steps += 1 # Conversion step needed if not GGUF
    elif skip_unet:
         total_steps = 1 + len(formats) # No extraction step

    completed = 0

    def _progress(stage):
        if progress_callback:
            progress_callback(completed, total_steps, stage)

    try:
        unet_path = unet_output
        f16_path = f16_output
        
        # 1. Extraction (or skipping)
        if not (skip_unet and skip_clips):
            _progress("Extracting Components")
            unet_path = extract_components(
                input_path,
                output_dir,
                model_name,
                overwrite,
                log_callback=log_callback,
                cancel_event=cancel_event,
                step_progress_callback=(
                    lambda frac, stage: progress_callback(completed + max(0.0, min(frac, 1.0)), total_steps, stage)
                    if progress_callback else None
                ),
                skip_unet=skip_unet,
                skip_clips=skip_clips
            )
            completed_paths.update([unet_output, clips_output])
            completed += 1
            _progress("Extracted Components")
        
        # 2. Conversion (if needed)
        # Check if input is GGUF, if so, skip conversion and use it as f16_path (renaming variable concept)
        if input_path.lower().endswith(".gguf") and skip_unet and skip_clips:
             f16_path = input_path
             _emit(f"Using input GGUF directly: {f16_path}", log_callback)
        else:
             # Need to convert unet -> GGUF
             # If skip_unet is True, input_path MIGHT BE the UNet safetensors itself if called from Tab 3.
             if skip_unet and skip_clips and input_path.lower().endswith(".safetensors"):
                 unet_path = input_path
                 _emit(f"Using input UNet directly: {unet_path}", log_callback)

             _progress("Converting to F16 GGUF")
             f16_path = convert_to_gguf(
                unet_path,
                output_dir,
                model_name,
                tools_dir,
                overwrite,
                log_callback=log_callback,
                cancel_event=cancel_event,
             )
             completed_paths.add(f16_output)
             completed += 1
             _progress("Converted to F16 GGUF")

        # 3. Quantization
        for fmt in formats:
            _check_cancel(cancel_event)
            _progress(f"Quantizing {fmt}")
            quantize_model(
                f16_path,
                output_dir,
                model_name,
                [fmt],
                tools_dir,
                overwrite,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )
            completed_paths.add(quant_outputs[fmt])
            completed += 1
            _progress(f"Completed {fmt}")

        return {
            "unet_path": unet_path,
            "f16_path": f16_path,
            "quantized_dir": os.path.join(output_dir, "quantized"),
            "model_name": model_name,
        }
    except PipelineCancelled:
        if cleanup_incomplete:
            _emit("Interrupted. Cleaning incomplete outputs...", log_callback)
            _cleanup_incomplete_paths(
                [path for path in tracked_paths if path not in completed_paths],
                preexisting_paths,
                log_callback=log_callback,
            )
        raise
    except Exception:
        if cleanup_incomplete:
            _emit("Error detected. Cleaning incomplete outputs...", log_callback)
            _cleanup_incomplete_paths(
                [path for path in tracked_paths if path not in completed_paths],
                preexisting_paths,
                log_callback=log_callback,
            )
        raise

def process_batch(
    input_dir,
    output_dir,
    formats=None,
    overwrite=False,
    tools_dir=None,
    log_callback=None,
    progress_callback=None,
    cancel_event=None,
):
    _check_cancel(cancel_event)
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".safetensors")]
    total_files = len(files)
    if total_files == 0:
        _emit("No .safetensors files found in input directory.", log_callback)
        return

    _emit(f"Found {total_files} models to process.", log_callback)
    
    for idx, filename in enumerate(files, start=1):
        _check_cancel(cancel_event)
        file_path = os.path.join(input_dir, filename)
        _emit(f"=== Processing batch item {idx}/{total_files}: {filename} ===", log_callback)
        
        try:
            process_model(
                input_path=file_path,
                output_dir=output_dir,
                formats=formats,
                overwrite=overwrite,
                tools_dir=tools_dir,
                log_callback=log_callback,
                progress_callback=None, # We don't track sub-progress in batch mode effectively yet
                cancel_event=cancel_event,
                cleanup_incomplete=True,
                skip_unet=False,
                skip_clips=False # Batch mode assumes full pipeline for now
            )
            _emit(f"=== Completed {filename} ===", log_callback)
        except PipelineCancelled:
            raise
        except Exception as e:
            _emit(f"ERROR processing {filename}: {e}", log_callback)
            _emit("Skipping to next model...", log_callback)
            continue
    
    _emit("Batch processing complete.", log_callback)


def main():
    parser = argparse.ArgumentParser(description="SDXL Extractor & Quantizer")
    parser.add_argument("--config", default="extraction_output_config.json", help="Path to config JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if they exist")
    args = parser.parse_args()
    
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        input_path = config.get("input")
        output_dir = config.get("output")
        formats = config.get("formats", ["Q8_0", "Q5_K_M", "Q4_K_M"])
        # Check if overwrite is also defined in JSON, otherwise use CLI flag
        overwrite = config.get("overwrite", args.overwrite)
    else:
        print(f"Error: Configuration file {args.config} not found.")
        return

    try:
        process_model(
            input_path=input_path,
            output_dir=output_dir,
            formats=formats,
            overwrite=overwrite,
            tools_dir=tools_dir,
        )
        print("\nAll components processed successfully!")
    except PipelineCancelled as e:
        print(f"\nInterrupted: {e}")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == "__main__":
    main()
