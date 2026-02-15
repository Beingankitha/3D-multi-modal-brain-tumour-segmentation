from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brain_tumor_segmentation",
    version="0.1.0",
    author="Research Team",
    description="3D multi-modal brain tumour segmentation with explainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Beingankitha/3D-multi-modal-brain-tumour-segmentation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "monai>=1.3.0",
        "numpy>=1.24.0",
        "nibabel>=5.0.0",
        "omegaconf>=2.3.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
)
