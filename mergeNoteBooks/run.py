import nbformat
import argparse
import os

def merge_notebooks(notebook_path1, notebook_path2, output_path):
    """
    Merges two Jupyter notebooks into a single notebook, preserving all cell content and outputs.

    Args:
        notebook_path1 (str): Path to the first Jupyter notebook (.ipynb file).
        notebook_path2 (str): Path to the second Jupyter notebook (.ipynb file).
        output_path (str): Path for the merged output Jupyter notebook (.ipynb file).
    """
    try:
        # Read the first notebook
        print(f"Reading first notebook: {notebook_path1}")
        with open(notebook_path1, 'r', encoding='utf-8') as f1:
            nb1 = nbformat.read(f1, as_version=4)

        # Read the second notebook
        print(f"Reading second notebook: {notebook_path2}")
        with open(notebook_path2, 'r', encoding='utf-8') as f2:
            nb2 = nbformat.read(f2, as_version=4)

        # The merged notebook will primarily use the metadata of the first notebook.
        # Append cells from the second notebook to the first one.
        # The 'outputs' are part of the cell structure and are preserved.
        nb1.cells.extend(nb2.cells)

        # Optional: Validate the merged notebook structure (can be helpful for debugging)
        # try:
        #     nbformat.validate(nb1)
        # except nbformat.ValidationError as e:
        #     print(f"Warning: Validation error in merged notebook - {e}")

        # Write the merged notebook to the output file
        print(f"Writing merged notebook to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            nbformat.write(nb1, f_out)
        
        print(f"Successfully merged '{os.path.basename(notebook_path1)}' and '{os.path.basename(notebook_path2)}' into '{os.path.basename(output_path)}'")

    except FileNotFoundError:
        print(f"Error: One or both notebook files not found. Please check the paths.")
        if not os.path.exists(notebook_path1):
            print(f"File not found: {notebook_path1}")
        if not os.path.exists(notebook_path2):
            print(f"File not found: {notebook_path2}")
    except Exception as e:
        print(f"An error occurred during merging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two Jupyter notebooks (.ipynb files) into a new notebook, preserving cell outputs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "notebook1", 
        help="Path to the first Jupyter notebook file (e.g., notebookA.ipynb)."
    )
    parser.add_argument(
        "notebook2", 
        help="Path to the second Jupyter notebook file (e.g., notebookB.ipynb)."
    )
    parser.add_argument(
        "output_notebook", 
        help="Path for the new merged Jupyter notebook file (e.g., merged_notebook.ipynb)."
    )

    args = parser.parse_args()

    # Basic validation for file extensions
    if not all(arg.endswith(".ipynb") for arg in [args.notebook1, args.notebook2, args.output_notebook]):
        parser.error("Error: All notebook file paths must end with .ipynb")
        # Note: parser.error() exits the script, so no further code in this block will run.

    # Ensure output directory exists if a full path is given
    output_dir = os.path.dirname(args.output_notebook)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}': {e}")
            # Exit if directory cannot be created, as write will fail.
            exit(1)
            
    merge_notebooks(args.notebook1, args.notebook2, args.output_notebook)