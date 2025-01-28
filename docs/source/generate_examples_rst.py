import os
import shutil


def clear_directory(directory):
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
    for nb in notebooks:
        source_path = os.path.join(examples_dir, f'{nb}.ipynb')
        target_path = os.path.join(target_dir, f'{nb}.ipynb')
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(source_path, target_path)


def find_notebooks(directory):
    """
    Returns a list of notebook paths (with no file extension),
    relative to 'directory'. For example, if a notebook is:
      examples/gLV/examples-bayes-gLV.ipynb
    we return 'gLV/examples-bayes-gLV' (forward slashes).
    """
    notebooks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, start=directory)
                notebooks.append(
                    relative_path.replace(
                        os.path.sep, '/').replace('.ipynb', '')
                )
    return notebooks


def group_notebooks_by_top_dir(notebooks):
    """
    Takes a list of relative notebook paths like ['gLV/examples-bayes-gLV', 'CRM/examples-sim-CRM']
    and returns a dict grouping them by the top-level folder:
      {
        'CRM': ['CRM/examples-sim-CRM'],
        'gLV': ['gLV/examples-bayes-gLV']
        ...
      }
    """
    grouped: dict[str, list[str]] = {}
    for nb in notebooks:
        parts = nb.split('/')
        top_dir = parts[0]  # The directory before the first slash
        grouped.setdefault(top_dir, []).append(nb)
    return grouped


def generate_rst(notebooks, target_dir, output_file):
    """
    Generates 'examples_auto.rst', grouping notebooks by their top-level folder.
    Each folder gets its own subheading and toctree.
    """
    grouped = group_notebooks_by_top_dir(notebooks)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Auto-Generated Notebook Listing\n")
        f.write("================================\n\n")

        # For each top-level folder, create a subheading and toctree
        # Sort the folder names so the output is consistent
        for folder in sorted(grouped.keys()):
            f.write(folder + "\n" + "-" * len(folder) + "\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n\n")

            # Write each notebook path
            for nb in sorted(grouped[folder]):
                # notebooks are now in docs/source/notebooks
                nb_path = os.path.join(
                    'notebooks', nb).replace(os.path.sep, '/')
                f.write(f"   {nb_path}\n")

            f.write("\n")  # blank line between sections


def main():
    # Path to your top-level examples directory
    examples_dir = os.path.abspath('../../examples')

    # The target dir in docs/source where notebooks will be copied
    target_dir = os.path.join(os.path.dirname(__file__), 'notebooks')

    # Clear the target directory before re-copying notebooks
    clear_directory(target_dir)

    notebooks = find_notebooks(examples_dir)

    # Copy all notebooks from examples/ to docs/source/notebooks/
    copy_notebooks(notebooks, examples_dir, target_dir)

    # Generate 'examples_auto.rst' (instead of 'examples.rst')
    output_rst = os.path.join(os.path.dirname(__file__), 'examples_auto.rst')
    generate_rst(notebooks, target_dir, output_rst)
    print(f"Generated {output_rst}")


if __name__ == "__main__":
    main()
