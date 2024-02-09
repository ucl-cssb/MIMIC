# generate_examples_rst.py
import os


def find_notebooks(directory):
    notebooks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, directory)
                notebooks.append(relative_path.replace(
                    os.path.sep, '/').replace('.ipynb', ''))
    return notebooks


# Adjust the examples_dir path if necessary
def generate_rst(notebooks, output_file):
    with open(output_file, 'w') as f:
        f.write("Examples\n")
        # Ensure the underline is at least as long as the title
        f.write("========\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        for nb in notebooks:
            # Adjust the path to be relative to the Sphinx source directory
            corrected_path = os.path.relpath(os.path.join(
                examples_dir, nb), start=os.path.dirname(output_file))
            f.write(
                f"   {corrected_path.replace(os.path.sep, '/').replace('.ipynb', '')}\n")


if __name__ == "__main__":
    # Adjust the path to your examples folder
    examples_dir = os.path.abspath('../../examples')
    notebooks = find_notebooks(examples_dir)
    output_rst = os.path.join(os.path.dirname(__file__), 'examples.rst')
    generate_rst(notebooks, output_rst)
    print(f"Generated {output_rst}")
