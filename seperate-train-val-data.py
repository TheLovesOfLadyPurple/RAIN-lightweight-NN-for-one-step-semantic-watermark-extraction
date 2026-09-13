import os
import shutil
from pathlib import Path
import glob
from collections import defaultdict

def cut_and_paste_files(source_base_dir, destination_base_dir, max_files=10000, file_extensions=('.pth', '.png'),iterative=False):
    """
    Cut and paste the first max_files from folders containing .pth and .png files.
    
    Args:
        source_base_dir (str): Base directory containing subfolders with .pth and .png files
        destination_base_dir (str): Destination directory where files will be moved
        max_files (int): Maximum number of files to process (default: 10000)
        file_extensions (tuple): File extensions to look for (default: ('.pth', '.png'))
    """
    
    source_path = Path(source_base_dir)
    dest_path = Path(destination_base_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if not source_path.exists():
        print(f"Source directory {source_base_dir} does not exist!")
        return
    
    # Find all files with specified extensions in all subdirectories
    all_files = []
    for ext in file_extensions:
        pattern = f"**/*{ext}"
        files = list(source_path.glob(pattern))
        all_files.extend(files)
    
    # Sort files by name to ensure consistent ordering
    all_files.sort(key=lambda x: x.name)
    
    # Group files by their parent directory to maintain folder structure
    files_by_dir = defaultdict(list)
    for file_path in all_files:
        relative_parent = file_path.parent.relative_to(source_path)
        files_by_dir[relative_parent].append(file_path)
    
    total_moved = 0
    moved_files = []
    
    print(f"Found {len(all_files)} total files with extensions {file_extensions}")
    print(f"Found {len(files_by_dir)} folders to process")
    print(f"Will move up to {max_files} files from each folder")
    print("-" * 50)
    
    # Process files - take max_files from each folder
    for folder, files in files_by_dir.items():
        # Create corresponding destination folder
        dest_folder = dest_path / folder
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Sort files in current folder
        files.sort(key=lambda x: x.name)
        
        # Take only the first max_files from this folder
        files_to_move = files[:max_files]
        
        print(f"Processing folder: {folder} ({len(files_to_move)}/{len(files)} files)")
        
        folder_moved = 0
        for file_path in files_to_move:
            # Destination file path
            dest_file = dest_folder / file_path.name
            
            try:
                # Cut (move) the file
                shutil.move(str(file_path), str(dest_file))
                moved_files.append((str(file_path), str(dest_file)))
                total_moved += 1
                folder_moved += 1
                
                if folder_moved % 100 == 0:  # Progress indicator for this folder
                    print(f"  Moved {folder_moved}/{len(files_to_move)} files from {folder}...")
                    
            except Exception as e:
                print(f"  Error moving {file_path}: {e}")
        
        print(f"  Completed {folder}: moved {folder_moved} files")
    
    print("-" * 50)
    print(f"Successfully moved {total_moved} files total")
    print(f"Files moved from: {source_base_dir}")
    print(f"Files moved to: {destination_base_dir}")
    
    # Save log of moved files
    log_file = dest_path / "moved_files_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Moved {total_moved} files\n")
        f.write(f"Source: {source_base_dir}\n")
        f.write(f"Destination: {destination_base_dir}\n")
        f.write("-" * 50 + "\n")
        for source, dest in moved_files:
            f.write(f"{source} -> {dest}\n")
    
    print(f"Log saved to: {log_file}")


def main():
    # Configuration - modify these paths according to your needs
    SOURCE_DIR = "./pd12m"
    DESTINATION_DIR = "./separated_files"
    MAX_FILES = 1000  # Files to move from EACH folder
    FILE_EXTENSIONS = ('.pth', '.jpg', '.png','.txt')
    
    print("File Separation Tool")
    print("=" * 50)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Destination directory: {DESTINATION_DIR}")
    print(f"Maximum files to move FROM EACH folder: {MAX_FILES}")
    print(f"File extensions: {FILE_EXTENSIONS}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Perform the cut and paste operation
    cut_and_paste_files(
        source_base_dir=SOURCE_DIR,
        destination_base_dir=DESTINATION_DIR,
        max_files=MAX_FILES,
        file_extensions=FILE_EXTENSIONS
    )


if __name__ == "__main__":
    main()
