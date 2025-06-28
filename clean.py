# IMPORTANT: READ BELOW






# IMPORTANT: make sure ModelNet40/processed/ does not exist (move it somewhere else)





import os
from tqdm import tqdm

def fix_off_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check if the first line needs fixing
    if lines[0].startswith("OFF") and len(lines[0].strip().split()) > 1:
        # Create corrected lines
        new_lines = ["OFF\n", lines[0][3:].strip() + '\n'] + lines[1:]

        # Overwrite the file with the corrected content
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True  # File was fixed
    return False  # File was already fine

def fix_all_off_files(root_dir):
    # Collect all .off file paths first
    off_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.off'):
                off_files.append(os.path.join(dirpath, filename))

    # Process with tqdm progress bar
    fixed_count = 0
    iterator = tqdm(off_files, desc="Fixing .off files", smoothing=0.0, dynamic_ncols=True)
    for file_path in iterator:
        iterator.set_description(file_path)
        if fix_off_file(file_path):
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count} / {len(off_files)}")

if __name__ == "__main__":
    root = 'data/uncompressed/ModelNet40/raw'
    fix_all_off_files(root)
