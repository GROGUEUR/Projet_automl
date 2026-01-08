"""
Script d'installation du paquet AutoML.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="automl",
    version="1.0.0",
    author="Comte Quentin, Lepine Francois, Floch Samuel, Delamare Bastien",
    description="Système AutoML pour l'entraînement et l'évaluation automatique de modèles ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GROGUEUR/Projet_automl.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3,<2.0.0",
        "pandas>=2.0.3,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "joblib>=1.3.2,<2.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.12.0,<1.0.0",
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
