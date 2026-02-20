import os
import zipfile

def create_release():
    release_name = "SDXL_GGUF_Quantize_Tool_Portable.zip"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(script_dir, release_name)

    # Directories and files to exclude from the release
    exclude_dirs = {'venv', '.git', '__pycache__', '.pytest_cache', 'outputs'}
    exclude_files = {release_name, 'build_release.py', '.gitignore', 'P3_technical_discovery_precision.md'}
    exclude_exts = {'.log'}

    print(f"Building portable release: {release_name}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(script_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files:
                    continue
                if any(file.endswith(ext) for ext in exclude_exts):
                    continue

                file_path = os.path.join(root, file)
                # Calculate relative path for zip architecture
                arcname = os.path.relpath(file_path, script_dir)
                
                print(f"Adding {arcname}...")
                zipf.write(file_path, arcname)

    print(f"\nSuccessfully built {release_name}!")
    print("Distribute this ZIP file to your users for a true 1-click portable experience.")

if __name__ == "__main__":
    create_release()
