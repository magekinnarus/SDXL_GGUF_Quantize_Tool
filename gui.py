import os
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import multiprocessing
import traceback

try:
    import customtkinter as ctk
except ImportError as exc:
    raise SystemExit("customtkinter is required. Install with: pip install customtkinter") from exc

from model_extractor import PipelineCancelled, process_model, extract_components, quantize_model, process_batch, validate_tools

def _worker_entry_point(target_name, args, event_queue, cancel_event):
    """Entry point for the background process. Bridging callbacks to the Queue."""
    try:
        # Resolve target function
        if target_name == "_run_full":
            func = process_model
            # Extraction needs skip_unet=False explicitly
            args["skip_unet"] = False
        elif target_name == "_run_extraction_only":
            func = extract_components
            # extract_components needs model_name if not provided
            if "model_name" not in args:
                model_name = os.path.splitext(os.path.basename(args["input_path"]))[0]
                # Strip common extraction suffixes (case-insensitive)
                for suffix in ["_unet", ".unet", "_f16", ".f16", "-f16"]:
                    if model_name.lower().endswith(suffix):
                        model_name = model_name[:-len(suffix)]
                args["model_name"] = model_name
            # Map input_path -> model_path
            if "input_path" in args:
                args["model_path"] = args.pop("input_path")
            if "output_dir" in args:
                args["output_dir"] = args.pop("output_dir")
        elif target_name == "_run_quant_only":
            func = process_model
            args["skip_unet"] = True
            args["skip_clips"] = True
        elif target_name == "_run_batch":
            func = process_batch
        else:
            raise ValueError(f"Unknown target: {target_name}")

        # Inject process-safe callbacks
        args['log_callback'] = lambda msg: event_queue.put(("log", msg))
        
        def _safe_progress(*p_args):
            if len(p_args) == 2: # (frac, stage)
                event_queue.put(("progress", p_args[0], 1.0, p_args[1]))
            elif len(p_args) >= 3: # (done, total, stage)
                event_queue.put(("progress", p_args[0], p_args[1], p_args[2]))
        
        if target_name == "_run_extraction_only":
            args['step_progress_callback'] = _safe_progress
        else:
            args['progress_callback'] = _safe_progress
            
        args['cancel_event'] = cancel_event
        
        # Tools dir injection
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        # Safely check if func accepts tools_dir
        real_func = getattr(func, "__func__", func)
        if hasattr(real_func, "__code__") and 'tools_dir' in real_func.__code__.co_varnames:
            args['tools_dir'] = tools_dir

        func(**args)
        event_queue.put(("done", "Completed successfully."))
    except PipelineCancelled:
        event_queue.put(("cancelled", "Process cancelled."))
    except Exception as e:
        event_queue.put(("error", f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"))


class SDXLQuantizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SDXL GGUF Quantize Tool")
        self.geometry("1100x800")
        self.minsize(950, 650)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # Variables for Tab 1 (Full)
        self.tab1_input_var = tk.StringVar()
        self.tab1_output_var = tk.StringVar()
        self.tab1_skip_clips_var = tk.BooleanVar(value=False)
        self.tab1_formats = {
            "Q8_0": tk.BooleanVar(value=True),
            "Q5_K_M": tk.BooleanVar(value=True),
            "Q4_K_M": tk.BooleanVar(value=True),
        }
        self.tab1_other_enabled = tk.BooleanVar(value=False)
        self.tab1_other_val = tk.StringVar()
        self.tab1_overwrite = tk.BooleanVar(value=False)

        # Variables for Tab 2 (Clips Only)
        # Variables for Tab 2 (Model Extraction Only)
        self.tab2_input_var = tk.StringVar()
        self.tab2_output_var = tk.StringVar()
        self.tab2_skip_unet_var = tk.BooleanVar(value=False)
        self.tab2_skip_clips_var = tk.BooleanVar(value=False)
        self.tab2_overwrite = tk.BooleanVar(value=False)

        # Variables for Tab 3 (Quantize Only)
        self.tab3_input_var = tk.StringVar()
        self.tab3_output_var = tk.StringVar()
        self.tab3_formats = {
            "Q8_0": tk.BooleanVar(value=True),
            "Q5_K_M": tk.BooleanVar(value=True),
            "Q4_K_M": tk.BooleanVar(value=True),
        }
        self.tab3_other_enabled = tk.BooleanVar(value=False)
        self.tab3_other_val = tk.StringVar()
        self.tab3_overwrite = tk.BooleanVar(value=False)

        # Variables for Tab 4 (Batch)
        self.tab4_input_var = tk.StringVar()
        self.tab4_output_var = tk.StringVar()
        self.tab4_formats = {
            "Q8_0": tk.BooleanVar(value=True),
            "Q5_K_M": tk.BooleanVar(value=True),
            "Q4_K_M": tk.BooleanVar(value=True),
        }
        self.tab4_other_enabled = tk.BooleanVar(value=False)
        self.tab4_other_val = tk.StringVar()
        self.tab4_overwrite = tk.BooleanVar(value=False)

        # Shared Status
        self.status_var = tk.StringVar(value="Idle")
        self.overall_progress_var = tk.StringVar(value="Progress: 0%")

        self.event_queue = multiprocessing.Queue()
        self.worker_process = None
        self.cancel_event = multiprocessing.Event()
        self.is_running = False

        self._build_layout()
        self.after(20, self._process_events) # Faster tick rate for better responsiveness

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1) # Tabs expand
        self.main_frame.grid_rowconfigure(2, weight=0) # Log expands

        # Tabs
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tabview.add("Full Pipeline")
        self.tabview.add("Model Extraction")
        self.tabview.add("Quantize Only")
        self.tabview.add("Batch Process")

        self._build_tab1(self.tabview.tab("Full Pipeline"))
        self._build_tab2(self.tabview.tab("Model Extraction"))
        self._build_tab3(self.tabview.tab("Quantize Only"))
        self._build_tab4(self.tabview.tab("Batch Process"))

        # Shared Progress & Log
        bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)
        
        status_frame = ctk.CTkFrame(bottom_frame)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(status_frame, text="Status:").grid(row=0, column=0, padx=(10, 5), pady=5)
        self.status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        self.overall_label = ctk.CTkLabel(status_frame, textvariable=self.overall_progress_var, anchor="e")
        self.overall_label.grid(row=0, column=2, padx=(5, 10), pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(bottom_frame)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.progress_bar.set(0.0)

        # Log
        ctk.CTkLabel(self.main_frame, text="Log Output:").grid(row=2, column=0, sticky="w", padx=12, pady=(0, 4))
        self.log_text = ctk.CTkTextbox(self.main_frame, height=180, wrap="word")
        self.log_text.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.log_text.configure(state="disabled")

        # Control Buttons
        btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        btn_frame.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 12))
        btn_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.run_button = ctk.CTkButton(btn_frame, text="Run Selected Tab", width=160, command=self._start_run)
        self.run_button.grid(row=0, column=0, padx=10)
        
        self.stop_button = ctk.CTkButton(
            btn_frame, text="Stop", width=160, command=self._stop_run,
            state="disabled", fg_color="#b33a3a", hover_color="#8d2c2c"
        )
        self.stop_button.grid(row=0, column=1, padx=10)

        self.reset_btn = ctk.CTkButton(btn_frame, text="Refresh UI", width=160, command=self._reset_ui_state)
        self.reset_btn.grid(row=0, column=2, padx=10, sticky="e")

    def _build_tab1(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        self._add_file_picker(parent, 0, "Input Model (.safetensors):", self.tab1_input_var, file=True)
        self._add_file_picker(parent, 1, "Output Folder:", self.tab1_output_var, file=False)
        self._add_quant_options(parent, 2, self.tab1_formats, self.tab1_other_enabled, self.tab1_other_val)
        
        opts_frame = ctk.CTkFrame(parent, fg_color="transparent")
        opts_frame.grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=10)
        ctk.CTkCheckBox(opts_frame, text="Overwrite Existing Files", variable=self.tab1_overwrite).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(opts_frame, text="Skip CLIPs Extraction", variable=self.tab1_skip_clips_var).pack(side="left")

    def _build_tab2(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        self._add_file_picker(parent, 0, "Input Model (.safetensors):", self.tab2_input_var, file=True)
        self._add_file_picker(parent, 1, "Output Folder:", self.tab2_output_var, file=False)
        
        opts_frame = ctk.CTkFrame(parent, fg_color="transparent")
        opts_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=12, pady=10)
        ctk.CTkCheckBox(opts_frame, text="Skip extracting UNet", variable=self.tab2_skip_unet_var).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(opts_frame, text="Skip extracting CLIPs", variable=self.tab2_skip_clips_var).pack(side="left", padx=(0, 20))
        ctk.CTkCheckBox(opts_frame, text="Overwrite Existing Files", variable=self.tab2_overwrite).pack(side="left")

    def _build_tab3(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        self._add_file_picker(parent, 0, "Input (F16.gguf OR UNet.safetensors):", self.tab3_input_var, file=True, ext="*.gguf *.safetensors")
        self._add_file_picker(parent, 1, "Output Folder:", self.tab3_output_var, file=False)
        self._add_quant_options(parent, 2, self.tab3_formats, self.tab3_other_enabled, self.tab3_other_val)
        ctk.CTkCheckBox(parent, text="Overwrite Existing Files", variable=self.tab3_overwrite).grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=10)

    def _build_tab4(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        self._add_file_picker(parent, 0, "Input Folder:", self.tab4_input_var, file=False)
        self._add_file_picker(parent, 1, "Output Folder:", self.tab4_output_var, file=False)
        self._add_quant_options(parent, 2, self.tab4_formats, self.tab4_other_enabled, self.tab4_other_val)
        ctk.CTkCheckBox(parent, text="Overwrite Existing Files", variable=self.tab4_overwrite).grid(row=3, column=0, columnspan=3, sticky="w", padx=12, pady=10)

    def _add_file_picker(self, parent, row, label, var, file=True, ext="*.*"):
        ctk.CTkLabel(parent, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=10)
        ctk.CTkEntry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=10)
        cmd = lambda: self._browse(var, file, ext)
        ctk.CTkButton(parent, text="Browse...", width=100, command=cmd).grid(row=row, column=2, padx=12, pady=10)

    def _add_quant_options(self, parent, row, formats_dict, other_enabled, other_val):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=8)
        frame.grid_columnconfigure(4, weight=1)
        ctk.CTkLabel(frame, text="Quantization:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        
        col = 1
        for name, var in formats_dict.items():
            ctk.CTkCheckBox(frame, text=name, variable=var).grid(row=0, column=col, padx=8, pady=8, sticky="w")
            col += 1
        
        ctk.CTkCheckBox(frame, text="Other", variable=other_enabled, command=lambda: self._toggle_other(other_enabled, entry)).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        entry = ctk.CTkEntry(frame, textvariable=other_val, state="disabled", placeholder_text="e.g. Q6_K")
        entry.grid(row=1, column=1, columnspan=4, sticky="ew", padx=8, pady=(0, 10))

    def _browse(self, var, file_mode, ext):
        if self.is_running: return
        if file_mode:
            path = filedialog.askopenfilename(filetypes=[("Model Files", ext), ("All Files", "*.*")])
        else:
            path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _toggle_other(self, var, entry):
        entry.configure(state="normal" if var.get() else "disabled")

    def _collect_formats(self, fmt_dict, other_enabled, other_val):
        fmts = [f for f, v in fmt_dict.items() if v.get()]
        if other_enabled.get():
            custom = [x.strip() for x in re.split(r"[\s,;]+", other_val.get()) if x.strip()]
            fmts.extend(custom)
        return list(set(fmts))

    def _start_run(self):
        if self.worker_process and self.worker_process.is_alive(): return
        
        tab_name = self.tabview.get()
        self.cancel_event.clear()
        
        target = None
        args = {}

        try:
            if tab_name == "Full Pipeline":
                input_path = self.tab1_input_var.get().strip()
                output_path = self.tab1_output_var.get().strip()
                formats = self._collect_formats(self.tab1_formats, self.tab1_other_enabled, self.tab1_other_val)
                if not input_path or not output_path: raise ValueError("Input and Output are required.")
                if not formats: raise ValueError("Select at least one format.")
                target = "_run_full"
                args = {
                    "input_path": input_path, "output_dir": output_path, "formats": formats,
                    "overwrite": self.tab1_overwrite.get(), "skip_clips": self.tab1_skip_clips_var.get()
                }

            elif tab_name == "Model Extraction":
                input_path = self.tab2_input_var.get().strip()
                output_path = self.tab2_output_var.get().strip()
                if not input_path or not output_path: raise ValueError("Input and Output are required.")
                target = "_run_extraction_only"
                args = {
                    "input_path": input_path, "output_dir": output_path,
                    "skip_unet": self.tab2_skip_unet_var.get(),
                    "skip_clips": self.tab2_skip_clips_var.get(),
                    "overwrite": self.tab2_overwrite.get()
                }

            elif tab_name == "Quantize Only":
                input_path = self.tab3_input_var.get().strip()
                output_path = self.tab3_output_var.get().strip()
                formats = self._collect_formats(self.tab3_formats, self.tab3_other_enabled, self.tab3_other_val)
                if not input_path or not output_path: raise ValueError("Input and Output are required.")
                if not formats: raise ValueError("Select at least one format.")
                target = "_run_quant_only"
                args = {
                    "input_path": input_path, "output_dir": output_path, "formats": formats,
                    "overwrite": self.tab3_overwrite.get()
                }

            elif tab_name == "Batch Process":
                input_path = self.tab4_input_var.get().strip()
                output_path = self.tab4_output_var.get().strip()
                formats = self._collect_formats(self.tab4_formats, self.tab4_other_enabled, self.tab4_other_val)
                if not input_path or not output_path: raise ValueError("Input and Output folders are required.")
                if not formats: raise ValueError("Select at least one format.")
                target = "_run_batch"
                args = {
                    "input_dir": input_path, "output_dir": output_path, "formats": formats,
                    "overwrite": self.tab4_overwrite.get()
                }
            
            self.is_running = True
            self.status_var.set(f"Running {tab_name}...")
            self.progress_bar.set(0.0)
            self.overall_progress_var.set("Progress: 0%")
            self.run_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            
            # Prepare arguments for multiprocessing
            # We must pass the target function name and its arguments, or a wrapper
            # Since self.target is a bound method in some cases, we'll use a standardized bridge.
            
            self.worker_process = multiprocessing.Process(
                target=_worker_entry_point,
                args=(target, args, self.event_queue, self.cancel_event),
                daemon=True
            )
            self.worker_process.start()

        except Exception as e:
            messagebox.showerror("Validation Error", str(e))


    # --- Stop Logic ---

    def _stop_run(self):
        self.cancel_event.set()
        self.status_var.set("Stopping...")
        self.stop_button.configure(state="disabled")
        
        # Give it a moment to stop gracefully, otherwise force it
        def _terminate_check():
            time.sleep(3)
            if self.is_running and self.worker_process and self.worker_process.is_alive():
                self.worker_process.terminate()
                self.event_queue.put(("log", "Force terminated process."))
                self.is_running = False
                self._reset_ui()
                self.status_var.set("Idle (Force Stopped)")
        
        threading.Thread(target=_terminate_check, daemon=True).start()

    # --- Event Loop ---

    def _process_events(self):
        try:
            log_batch = []
            for _ in range(100):
                try:
                    ev = self.event_queue.get_nowait()
                except queue.Empty:
                    break

                dtype = ev[0]
                if dtype == "log":
                    log_batch.append(ev[1])
                elif dtype == "progress":
                    done, total, stage = ev[1], ev[2], ev[3]
                    frac = min(max(done / max(total, 1), 0.0), 1.0)
                    self.progress_bar.set(frac)
                    current_step = int(done) + 1 if int(done) < int(total) else int(total)
                    if current_step > 0 and int(total) > 0:
                        self.overall_progress_var.set(f"Step {current_step} of {int(total)}: {stage}")
                    else:
                        self.overall_progress_var.set(str(stage))
                elif dtype == "done":
                    self.is_running = False
                    self.status_var.set("Done")
                    self._reset_ui()
                    messagebox.showinfo("Success", ev[1])
                elif dtype == "cancelled":
                    self.is_running = False
                    self.status_var.set("Cancelled")
                    self._reset_ui()
                    messagebox.showinfo("Cancelled", ev[1])
                elif dtype == "error":
                    self.is_running = False
                    self.status_var.set("Error")
                    self._reset_ui()
                    messagebox.showerror("Error", ev[1])

            if log_batch:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", "\n".join(log_batch) + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")

        except Exception as e:
            # Fallback to prevent loop death
            print(f"Error in UI event loop: {e}")
        
        self.after(20, self._process_events)

    def _reset_ui(self):
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _stop_run(self):
        self.cancel_event.set()
        self.status_var.set("Stopping...")
        self.stop_button.configure(state="disabled")

    def _reset_ui_state(self):
        if self.is_running: return
        self.status_var.set("Idle")
        self.progress_bar.set(0.0)
        self.overall_progress_var.set("Progress: 0%")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = SDXLQuantizerApp()
    app.mainloop()
