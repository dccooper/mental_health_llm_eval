# setup.py — Package definition for mental_health_llm_eval
#
# This file defines how to package, install, and expose functionality from the
# mental health LLM evaluation toolkit. It allows others to install this as a
# module, use it in Python code, or run the CLI tool from the terminal.

from setuptools import setup, find_packages

# Call to setuptools.setup() configures the package metadata and installation behavior
setup(
    # The name of the package users will install (e.g., pip install mental_health_llm_eval)
    name='mental_health_llm_eval',

    # Version of the package; helps manage updates and compatibility
    version='0.1.0',

    # Short description of the package's purpose
    description='Evaluation framework for mental health-related large language models',

    # Author information (replace with your own)
    author='Your Name',
    author_email='your@email.com',

    # URL to the repository or project homepage
    url='https://github.com/YOUR_USERNAME/mental_health_llm_eval',

    # This tells setuptools to include all Python modules in the `src` directory
    packages=find_packages(include=['src', 'src.*']),

    # External packages required to run this project
    install_requires=[
        'pandas',
        'pyyaml',
        'streamlit',
    ],

    # Define command-line tools the user can run after installing this package
    entry_points={
        'console_scripts': [
            # This means a user can run `mh-eval` from the terminal and it will call cli/main.py:main()
            'mh-eval=cli.main:main',
        ],
    },

    # Classification tags used by PyPI and other tools to understand the package
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

    # The minimum Python version required
    python_requires='>=3.8',
)
