from setuptools import setup, find_packages

setup(
    name="automl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'joblib>=1.3.0',
    ],
    author="Votre Groupe",
    description="AutoML package pour le challenge Machine Learning",
    python_requires='>=3.8',
)
