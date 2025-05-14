"""
Mental Health LLM Evaluator
=========================

A framework for evaluating LLM responses in mental health contexts.
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Call to setuptools.setup() configures the package metadata and installation behavior
setup(
    # The name of the package users will install (e.g., pip install mental_health_llm_eval)
    name='mental_health_llm_eval',

    # Version of the package; helps manage updates and compatibility
    version='0.1.0',

    # Short description of the package's purpose
    description='Framework for evaluating LLM responses in mental health contexts',

    # Long description of the package
    long_description=long_description,

    # Type of long description
    long_description_content_type='text/markdown',

    # Author information (replace with your own)
    author='Your Name',
    author_email='your.email@example.com',

    # URL to the repository or project homepage
    url='https://github.com/yourusername/mental_health_llm_eval',

    # This tells setuptools to include all Python modules in the `src` directory
    packages=find_packages(where='src'),

    # The directory where the package's __init__.py files are located
    package_dir={'': 'src'},

    # External packages required to run this project
    install_requires=requirements,

    # Additional dependencies for development, documentation, UI, and models
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
            'flake8>=6.0.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
            'sphinx-autodoc-typehints>=1.24.0',
        ],
        'ui': [
            'streamlit>=1.29.0',
            'plotly>=5.18.0',
        ],
        'models': {
            'openai': ['openai>=1.0.0'],
            'anthropic': ['anthropic>=0.8.1'],
            'huggingface': [
                'transformers>=4.36.0',
                'torch>=2.1.0',
                'accelerate>=0.25.0',
                'bitsandbytes>=0.41.0',
            ],
        }
    },

    # Define command-line tools the user can run after installing this package
    entry_points={
        'console_scripts': [
            # This means a user can run `mh-llm-eval` from the terminal and it will call mental_health_llm_eval.cli:main
            'mh-llm-eval=mental_health_llm_eval.cli:main',
            'mh-llm-eval-ui=mental_health_llm_eval.ui:main',
        ],
    },

    # Classification tags used by PyPI and other tools to understand the package
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Healthcare',
    ],

    # Keywords for the package
    keywords='llm, mental health, evaluation, nlp, machine learning',

    # Additional project URLs
    project_urls={
        'Documentation': 'https://mental-health-llm-eval.readthedocs.io/',
        'Source': 'https://github.com/yourusername/mental_health_llm_eval',
        'Issues': 'https://github.com/yourusername/mental_health_llm_eval/issues',
    },

    # The minimum Python version required
    python_requires='>=3.8',
)
