import os
import shutil


def clear_directory(directory):
    """Clears the given directory, deleting all files and subfolders."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(directory, exist_ok=True)


def copy_notebooks(notebooks, examples_dir, target_dir):
    """
    Copies each notebook from the examples directory to the target directory.
    The `notebooks` list contains paths relative to `examples_dir` (without the .ipynb extension).
    """
    for nb in notebooks:
        source_path = os.path.join(examples_dir, f'{nb}.ipynb')
        target_path = os.path.join(target_dir, f'{nb}.ipynb')
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)


def find_notebooks(directory):
    """
    Returns a list of notebook paths (with no file extension),
    relative to `directory`. For example, if a notebook is:
      examples/gLV/examples-bayes-gLV.ipynb
    this function returns 'gLV/examples-bayes-gLV' (using forward slashes).
    """
    notebooks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, start=directory)
                notebooks.append(relative_path.replace(
                    os.path.sep, '/').replace('.ipynb', ''))
    return notebooks


def group_notebooks_by_top_dir(notebooks):
    """
    Groups a list of relative notebook paths (e.g. 'gLV/examples-bayes-gLV')
    by their top-level folder.
    Returns a dictionary, for example:
      {
          'CRM': ['CRM/examples-sim-CRM'],
          'gLV': ['gLV/examples-bayes-gLV', 'gLV/examples-sim-gLV'],
          ...
      }
    """
    grouped: dict[str, list[str]] = {}
    for nb in notebooks:
        parts = nb.split('/')
        top_dir = parts[0]
        grouped.setdefault(top_dir, []).append(nb)
    return grouped


def generate_rst(notebooks, output_file):
    """
    Generates a single 'examples.rst' file that includes a top-level header and,
    for each model group, a section with a toctree of the notebooks.
    """
    grouped = group_notebooks_by_top_dir(notebooks)

    # Mapping from folder name to a more descriptive title
    MODEL_NAMES = {
        "CRM": "Consumer Resource Model (CRM)",
        "gLV": "Generalized Lotka-Volterra (gLV)",
        "gMLV": "Generalized Metabolic Lotka-Volterra (gMLV)",
        "GP": "Gaussian Processes (GP)",
        "MVAR": "Multivariate Autoregressive Model (MVAR)",
        "VAR": "Vector Autoregression (VAR)",
        "MultiModel": "Multi-Model Analysis"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the top-level headers
        f.write("Examples\n")
        f.write("========\n\n")
        f.write("Jupyter Notebook Examples by Model\n")
        f.write("------------------------------------\n\n")

        # For each top-level folder, generate a section with a toctree.
        for folder in sorted(grouped.keys()):
            title = MODEL_NAMES.get(folder, folder)
            # Use a level-3 header (using '~' as the underline character)
            f.write(f"{title}\n")
            f.write("~" * len(title) + "\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n\n")
            for nb in sorted(grouped[folder]):
                # Assume notebooks are in the 'notebooks' subfolder relative to the .rst file.
                nb_path = os.path.join(
                    "notebooks", nb).replace(os.path.sep, '/')
                f.write(f"   {nb_path}\n")
            f.write("\n")


def main():
    # Path to your top-level examples directory (adjust relative path as needed)
    examples_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../examples"))

    # The target directory in docs/source where notebooks will be copied
    target_dir = os.path.join(os.path.dirname(__file__), "notebooks")

    # Clear the target directory before re-copying notebooks
    clear_directory(target_dir)

    # Find all notebooks in the examples directory
    notebooks = find_notebooks(examples_dir)

    # Copy notebooks from examples/ to docs/source/notebooks/
    copy_notebooks(notebooks, examples_dir, target_dir)

    # Generate 'examples.rst' in the same directory as this script
    output_rst = os.path.join(os.path.dirname(__file__), "examples.rst")
    generate_rst(notebooks, output_rst)
    print(f"Generated {output_rst}")


if __name__ == "__main__":
    main()
