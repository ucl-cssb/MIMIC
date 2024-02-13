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
    notebooks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, start=directory)
                notebooks.append(relative_path.replace(
                    os.path.sep, '/').replace('.ipynb', ''))
    return notebooks


def generate_rst(notebooks, target_dir, output_file):
    with open(output_file, 'w') as f:
        f.write("Examples\n")
        f.write("========\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        for nb in notebooks:
            # Notebooks are now within `docs/source/notebooks`, adjust path accordingly
            nb_path = os.path.join('notebooks', nb).replace(os.path.sep, '/')
            f.write(f"   {nb_path}\n")


def main():
    examples_dir = os.path.abspath('../../examples')
    # Target directory within docs/source
    target_dir = os.path.join(os.path.dirname(__file__), 'notebooks')

    # Clear the target directory before copying notebooks
    clear_directory(target_dir)

    notebooks = find_notebooks(examples_dir)

    # Copy notebooks to the Sphinx source directory
    copy_notebooks(notebooks, examples_dir, target_dir)

    # Now, generate examples.rst with paths relative to the new location
    output_rst = os.path.join(os.path.dirname(__file__), 'examples.rst')
    # Ensure paths are relative to `docs/source/notebooks`
    generate_rst(notebooks, target_dir, output_rst)
    print(f"Generated {output_rst}")


if __name__ == "__main__":
    main()
