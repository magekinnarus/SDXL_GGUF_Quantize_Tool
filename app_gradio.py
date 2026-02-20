import os
import gradio as gr
import traceback
import platform

from model_extractor import PipelineCancelled, process_model, extract_components, process_batch

import queue
import threading
import time

# Helper generator to yield progress to Gradio
def gradio_progress_callback(progress_obj, done, total, stage):
    frac = min(max(done / max(total, 1), 0.0), 1.0)
    current_step = int(done) + 1 if int(done) < int(total) else int(total)
    
    if current_step > 0 and int(total) > 0:
        desc = f"Step {current_step} of {int(total)}: {stage}"
    else:
        desc = str(stage)
        
    progress_obj(frac, desc=desc)

def parse_formats(q8, q5, q4, other_enabled, other_val):
    formats = []
    if q8: formats.append("Q8_0")
    if q5: formats.append("Q5_K_M")
    if q4: formats.append("Q4_K_M")
    
    if other_enabled and other_val:
        import re
        custom = [x.strip() for x in re.split(r"[\s,;]+", other_val) if x.strip()]
        formats.extend(custom)
    return list(set(formats))

def _run_with_queue(worker_func, progress):
    q = queue.Queue()
    log_output = []
    
    def log_cb(msg):
        q.put(("log", msg))
        
    def prog_cb(d, t, s):
        q.put(("progress", d, t, s))
        
    def worker():
        try:
            worker_func(log_cb, prog_cb)
            q.put(("done", "Process Completed Successfully!"))
        except Exception as e:
            q.put(("error", f"Error: {str(e)}\n\n{traceback.format_exc()}"))
            
    threading.Thread(target=worker, daemon=True).start()
    
    while True:
        try:
            msg_type, *args = q.get(timeout=0.1)
            if msg_type == "log":
                log_output.append(args[0])
                yield "\n".join(log_output)
            elif msg_type == "progress":
                gradio_progress_callback(progress, args[0], args[1], args[2])
            elif msg_type == "done":
                log_output.append(f"\n\nSuccess: {args[0]}")
                yield "\n".join(log_output)
                break
            elif msg_type == "error":
                log_output.append(f"\n\n{args[0]}")
                yield "\n".join(log_output)
                break
        except queue.Empty:
            yield "\n".join(log_output)
            continue

def run_full_pipeline(input_path, output_dir, q8, q5, q4, other_enabled, other_val, overwrite, skip_clips, progress=gr.Progress()):
    if not input_path or not output_dir:
        yield "Error: Input and Output paths must be provided."
        return
        
    formats = parse_formats(q8, q5, q4, other_enabled, other_val)
    if not formats:
        yield "Error: Please select at least one quantization format."
        return
        
    def _worker(log_cb, prog_cb):
        process_model(
            input_path=input_path, output_dir=output_dir, formats=formats, overwrite=overwrite,
            skip_clips=skip_clips, skip_unet=False, log_callback=log_cb, progress_callback=prog_cb, tools_dir=None
        )
    yield from _run_with_queue(_worker, progress)

def run_extraction_only(input_path, output_dir, skip_unet, skip_clips, overwrite, progress=gr.Progress()):
    if not input_path or not output_dir:
        yield "Error: Input and Output paths must be provided."
        return
        
    model_name = os.path.splitext(os.path.basename(input_path))[0]
    for suffix in ["_unet", ".unet", "_f16", ".f16", "-f16"]:
        if model_name.lower().endswith(suffix):
            model_name = model_name[:-len(suffix)]
                
    def _worker(log_cb, prog_cb):
        extract_components(
            model_path=input_path, output_dir=output_dir, model_name=model_name,
            skip_unet=skip_unet, skip_clips=skip_clips, overwrite=overwrite,
            log_callback=log_cb, step_progress_callback=lambda f, s: prog_cb(f * 100, 100, s), tools_dir=None
        )
    yield from _run_with_queue(_worker, progress)

def run_quantize_only(input_path, output_dir, q8, q5, q4, other_enabled, other_val, overwrite, progress=gr.Progress()):
    if not input_path or not output_dir:
        yield "Error: Input and Output paths must be provided."
        return
        
    formats = parse_formats(q8, q5, q4, other_enabled, other_val)
    if not formats:
        yield "Error: Please select at least one quantization format."
        return
        
    def _worker(log_cb, prog_cb):
        process_model(
            input_path=input_path, output_dir=output_dir, formats=formats, overwrite=overwrite,
            skip_clips=True, skip_unet=True, log_callback=log_cb, progress_callback=prog_cb, tools_dir=None
        )
    yield from _run_with_queue(_worker, progress)

def run_batch_process(input_dir, output_dir, q8, q5, q4, other_enabled, other_val, overwrite, progress=gr.Progress()):
    if not input_dir or not output_dir:
        yield "Error: Input and Output directories must be provided."
        return
        
    formats = parse_formats(q8, q5, q4, other_enabled, other_val)
    if not formats:
        yield "Error: Please select at least one quantization format."
        return
        
    def _worker(log_cb, prog_cb):
        process_batch(
            input_dir=input_dir, output_dir=output_dir, formats=formats, overwrite=overwrite,
            log_callback=log_cb, progress_callback=prog_cb, tools_dir=None
        )
    yield from _run_with_queue(_worker, progress)



# -------------------------------------------------------------------------
# Gradio UI Construction
# -------------------------------------------------------------------------

with gr.Blocks(title="SDXL GGUF Quantize Tool") as demo:
    gr.Markdown("# SDXL GGUF Quantize Tool")
    gr.Markdown("Run the inference quantization pipeline cleanly from Colab!")
    
    with gr.Tabs():
        # Tab 1: Full Pipeline
        with gr.TabItem("Full Pipeline"):
            with gr.Row():
                t1_input = gr.Textbox(label="Input Model Path (.safetensors)")
                t1_output = gr.Textbox(label="Output Folder Path")
            
            with gr.Row():
                gr.Markdown("### Quantization Formats")
            with gr.Row():
                t1_q8 = gr.Checkbox(label="Q8_0", value=True)
                t1_q5 = gr.Checkbox(label="Q5_K_M", value=True)
                t1_q4 = gr.Checkbox(label="Q4_K_M", value=True)
            with gr.Row():
                t1_other_enabled = gr.Checkbox(label="Other Formats")
                t1_other_val = gr.Textbox(label="Comma-separated (e.g., Q6_K)", visible=False)
                
                # Toggle visibility of other textbox based on checkbox
                t1_other_enabled.change(fn=lambda x: gr.update(visible=x), inputs=t1_other_enabled, outputs=t1_other_val)
                
            with gr.Row():
                t1_skip_clips = gr.Checkbox(label="Skip CLIPs Extraction", value=False)
                t1_overwrite = gr.Checkbox(label="Overwrite Existing Files", value=False)
                
            t1_btn = gr.Button("Run Full Pipeline", variant="primary")
            t1_log = gr.Textbox(label="Log Output", lines=10, interactive=False)
            
            t1_btn.click(
                fn=run_full_pipeline,
                inputs=[t1_input, t1_output, t1_q8, t1_q5, t1_q4, t1_other_enabled, t1_other_val, t1_overwrite, t1_skip_clips],
                outputs=t1_log
            )

        # Tab 2: Model Extraction 
        with gr.TabItem("Model Extraction Only"):
            with gr.Row():
                t2_input = gr.Textbox(label="Input Model Path (.safetensors)")
                t2_output = gr.Textbox(label="Output Folder Path")
                
            with gr.Row():
                t2_skip_unet = gr.Checkbox(label="Skip UNet Extraction", value=False)
                t2_skip_clips = gr.Checkbox(label="Skip CLIPs Extraction", value=False)
                t2_overwrite = gr.Checkbox(label="Overwrite Existing Files", value=False)
                
            t2_btn = gr.Button("Run Extraction", variant="primary")
            t2_log = gr.Textbox(label="Log Output", lines=10, interactive=False)
            
            t2_btn.click(
                fn=run_extraction_only,
                inputs=[t2_input, t2_output, t2_skip_unet, t2_skip_clips, t2_overwrite],
                outputs=t2_log
            )

        # Tab 3: Quantize Only
        with gr.TabItem("Quantize Only"):
            with gr.Row():
                t3_input = gr.Textbox(label="Input Model Path (F16.gguf or UNet.safetensors)")
                t3_output = gr.Textbox(label="Output Folder Path")
                
            with gr.Row():
                gr.Markdown("### Quantization Formats")
            with gr.Row():
                t3_q8 = gr.Checkbox(label="Q8_0", value=True)
                t3_q5 = gr.Checkbox(label="Q5_K_M", value=True)
                t3_q4 = gr.Checkbox(label="Q4_K_M", value=True)
            with gr.Row():
                t3_other_enabled = gr.Checkbox(label="Other Formats")
                t3_other_val = gr.Textbox(label="Comma-separated (e.g., Q6_K)", visible=False)
                
                t3_other_enabled.change(fn=lambda x: gr.update(visible=x), inputs=t3_other_enabled, outputs=t3_other_val)
                
            with gr.Row():
                t3_overwrite = gr.Checkbox(label="Overwrite Existing Files", value=False)
                
            t3_btn = gr.Button("Run Quantization", variant="primary")
            t3_log = gr.Textbox(label="Log Output", lines=10, interactive=False)
            
            t3_btn.click(
                fn=run_quantize_only,
                inputs=[t3_input, t3_output, t3_q8, t3_q5, t3_q4, t3_other_enabled, t3_other_val, t3_overwrite],
                outputs=t3_log
            )

        # Tab 4: Batch Process
        with gr.TabItem("Batch Process"):
            with gr.Row():
                t4_input = gr.Textbox(label="Input Folder Path")
                t4_output = gr.Textbox(label="Output Folder Path")
                
            with gr.Row():
                gr.Markdown("### Quantization Formats")
            with gr.Row():
                t4_q8 = gr.Checkbox(label="Q8_0", value=True)
                t4_q5 = gr.Checkbox(label="Q5_K_M", value=True)
                t4_q4 = gr.Checkbox(label="Q4_K_M", value=True)
            with gr.Row():
                t4_other_enabled = gr.Checkbox(label="Other Formats")
                t4_other_val = gr.Textbox(label="Comma-separated (e.g., Q6_K)", visible=False)
                
                t4_other_enabled.change(fn=lambda x: gr.update(visible=x), inputs=t4_other_enabled, outputs=t4_other_val)
                
            with gr.Row():
                t4_overwrite = gr.Checkbox(label="Overwrite Existing Files", value=False)
                
            t4_btn = gr.Button("Run Batch Process", variant="primary")
            t4_log = gr.Textbox(label="Log Output", lines=10, interactive=False)
            
            t4_btn.click(
                fn=run_batch_process,
                inputs=[t4_input, t4_output, t4_q8, t4_q5, t4_q4, t4_other_enabled, t4_other_val, t4_overwrite],
                outputs=t4_log
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False, theme=gr.themes.Soft())
