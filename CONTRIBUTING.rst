===================================
Contributing to MIMIC
===================================

We're excited to have you contribute to MIMIC! This document outlines the guidelines for contributing to our project. By following these guidelines, you help us maintain the quality of our project and ensure that everyone's contributions can be incorporated smoothly.

Maintainer Handoff & Onboarding
-------------------------------

If you are taking over as a maintainer, please:

- Add your name and contact to AUTHORS.rst.
- Review HISTORY.rst for recent changes and handoff notes.
- Familiarize yourself with the codebase and documentation structure.
- Update this file with any new processes or guidelines as the project evolves.

Getting Started
----------------

Before you begin:

* Make sure you have a `GitHub account <https://github.com/signup/free>`_.
* Familiarize yourself with the `GitHub flow <https://guides.github.com/introduction/flow/>`_.
* Check the issues for any features or bugs you might be interested in, or create a new issue to discuss your own ideas.

Making Changes
----------------

#. **Fork the Repository**
   * Fork the repository on GitHub.

#. **Clone the Fork**
   * Clone your fork to your local machine.

#. **Create a Branch**
   * Create a new branch for your changes.
   * Name the branch something descriptive (e.g., `add-comment-feature`).

#. **Make Your Changes**
   * Follow the coding standards and write quality code.
   * Add comments to your code where necessary.
   * Write tests for your changes.

#. **Commit Your Changes**
   * Make sure your commit messages are clear and descriptive.

#. **Push to the Branch**
   * Push your changes to your branch in your forked repository.

#. **Submit a Pull Request**
   * Open a pull request to the main branch of the original repository.
   * Ensure all actions in the PR checklist are completed before submitting.

Pull Request Checklist
------------------------

Before submitting your Pull Request, ensure:

* Your branch is re-based on the latest main branch.
* Necessary documentation is updated or added.
* Your code is well-commented, especially in complex or critical areas.
* Your code has been thoroughly tested in your local setup.
* The PR template questions are filled out to provide context on your changes.

Development Workflow
----------------------

Our GitHub Actions workflow ensures that every push to `master` and every pull request undergoes rigorous checks to maintain the quality and integrity of the code. Here's what happens under the hood:

1. **Automatic Dependency Management**: When changes are pushed to the repository, our workflow automatically updates the `requirements.txt` file to reflect the latest dependencies needed for the project. This ensures that the project always has up-to-date dependencies.

2. **Continuous Integration Tests**:
    - **Build**: The code is built across multiple operating systems (Ubuntu, macOS, and Windows) using different versions of Python to ensure cross-platform compatibility.
    - **Code Formatting**: We use `autopep8` to automatically format the code to adhere to PEP 8 standards. This helps in maintaining the readability and consistency of the code.
    - **Linting**: The `flake8` tool checks for syntax errors or undefined names in the code. It's crucial for catching common errors that might otherwise go unnoticed.
    - **Type Checking**: `mypy` is used for type checking, ensuring that type hints are used correctly throughout the codebase.
    - **Testing**: Our tests are run with `pytest`, which not only checks for correct behavior but also ensures that any new changes do not break existing functionalities.

3. **Handling Artifacts**: If the `requirements.txt` is updated, it's uploaded as an artifact to GitHub, which can be used for debugging or direct download from the Actions log.

**Why Your PR Might Fail**:

- **Dependency Issues**: If `requirements.txt` does not correctly reflect the project's dependencies due to an incomplete update.
- **Code Non-compliance**: Failing to meet formatting guidelines, having syntax errors, or failing type checks.
- **Test Failures**: Breaking changes that cause unit tests to fail.

Please ensure your code adheres to the guidelines laid out in this document and passes all checks before submitting a pull request.

Code Review Process
---------------------

#. **Reviewer Assignment**
   * Once you open a PR, a maintainer will assign a reviewer.

#. **Addressing Feedback**
   * If your PR receives comments, address them promptly.
   * Make any requested changes and push them to your branch.

#. **Approval and Merge**
   * After your PR is approved by a reviewer, a maintainer will merge it.

Reporting Bugs
----------------

* Use the GitHub issues to report bugs.
* Before creating a new issue, check if it has already been reported.
* Include detailed instructions on how to reproduce the bug.
* Add Tags here when implemented.

Suggesting Enhancements
-------------------------

* Use GitHub issues to suggest enhancements.
* Be clear and detailed in your suggestion.

Common Issues:
----------------
- My example doesn't show in documentation. Make sure that your Jupter notebook is in the correct folder, and more importantly, *that it has a title!*

Additional Notes
-----------------

Thank you for contributing to MIMIC!
