"""
Script d'installation du paquet AutoML.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="automl",
    version="0.1.0",
    author="Équipe AutoML M1 Info IA",
    author_email="",
    description="Système AutoML pour l'entraînement et l'évaluation automatique de modèles ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-repo/automl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Possibilité d'ajouter des scripts CLI ici
        ],
    },
)
