import os


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


def generate_rst(notebooks, examples_dir, output_file):
    with open(output_file, 'w') as f:
        f.write("Examples\n")
        f.write("========\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        for nb in notebooks:
            # Here, we adjust the path to make it correctly relative to the Sphinx source directory
            # Since the notebooks are in a parallel structure to the docs/source, we adjust accordingly
            # Assuming the examples directory is directly above the docs/source directory in the structure
            corrected_path = nb
            f.write(f"   ../examples/{corrected_path}\n")


def main():
    # Path to your examples folder, adjusted to be absolute
    examples_dir = os.path.abspath('../../examples')
    notebooks = find_notebooks(examples_dir)
    output_rst = os.path.join(os.path.dirname(__file__), 'examples.rst')
    generate_rst(notebooks, examples_dir, output_rst)
    print(f"Generated {output_rst}")


if __name__ == "__main__":
    main()
