import os
import zipfile

src_dir = r"d:\project\FYP_new"
out_zip = r"d:\project\FYP_new_Final.zip"

# Excluded folders (absolute paths)
exclude_dirs = [
    os.path.join(src_dir, 'venv'),
    os.path.join(src_dir, 'datasets'),
]

# We will also skip __pycache__ and .git entirely
skip_names = {'__pycache__', '.git', '.pytest_cache'}

def should_exclude(root):
    # Check if this root is within any of the excluded dirs
    for exc in exclude_dirs:
        if root == exc or root.startswith(exc + os.sep):
            return True
    return False

print(f"Zipping {src_dir} to {out_zip}...")
try:
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):
            if should_exclude(root):
                continue
                
            # Filter directories going forward
            dirs[:] = [d for d in dirs if d not in skip_names and not should_exclude(os.path.join(root, d))]
            
            for file in files:
                # Exclude the zip file itself if it's placed inside the source dir
                src_path = os.path.join(root, file)
                if src_path == out_zip:
                    continue
                
                arcname = os.path.relpath(src_path, start=src_dir)
                zipf.write(src_path, arcname)

    print(f"Successfully created zip archive at: {out_zip}")
except Exception as e:
    print(f"Error creating zip archive: {e}")
