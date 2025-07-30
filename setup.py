"""
Setup script for CTEntropy Diagnostic Platform.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ctentropy-platform",
    version="0.1.0",
    author="Betti Labs",
    author_email="contact@bettilabs.com",
    description="Symbolic entropy framework for early detection of neurological degeneration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bettilabs/ctentropy-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "full": [
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "mne>=1.0.0",
            "nibabel>=3.2.0",
            "fastapi>=0.70.0",
            "sqlalchemy>=1.4.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ctentropy=ctentropy_platform.cli:main",
        ],
    },
)