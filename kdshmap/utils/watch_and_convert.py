import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import jupytext
import nbformat

class NotebookHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.modified_files = set()  # Set to track modified files

    def on_modified(self, event):
        # Add modified notebook to the set if it's an .ipynb file
        if event.src_path.endswith(".ipynb"):
            self.modified_files.add(event.src_path)

def is_valid_notebook(notebook_path):
    """
    Check if the given file is a valid Jupyter Notebook file.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nbformat.read(f, as_version=4)
        return True
    except (nbformat.reader.NotJSONError, FileNotFoundError, PermissionError) as e:
        print(f"Skipping {notebook_path}: {e}")
        return False

def convert_notebooks(modified_files):
    # Convert each modified notebook to .py
    for notebook_path in list(modified_files):
        # Validate the notebook before conversion
        if not is_valid_notebook(notebook_path):
            modified_files.remove(notebook_path)
            continue
        
        script_path = notebook_path.replace(".ipynb", ".py")
        try:
            # Convert notebook to script
            jupytext.write(jupytext.read(notebook_path), script_path)
            print(f"Converted {notebook_path} to {script_path}")
            modified_files.remove(notebook_path)
        except Exception as e:
            print(f"Error converting {notebook_path}: {e}")

if __name__ == "__main__":
    path = "/home/nrohan/main/kdshmap/kdshmap/utils" # Update this path

    # Check if the directory exists
    if not os.path.exists(path):
        print(f"Error: The directory {path} does not exist.")
        exit(1)

    event_handler = NotebookHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    
    try:
        observer.start()
        print(f"Watching {path} for changes...")

        while True:
            # Convert modified notebooks every 10 seconds
            convert_notebooks(event_handler.modified_files)
            time.sleep(10)  # Change interval as needed
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
