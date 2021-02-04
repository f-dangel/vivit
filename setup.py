"""Setup file for lowrank."""

from setuptools import find_packages, setup

setup(
    author="Felix Dangel",
    name="lowrank",
    version="0.0.1",
    description="Use low-rank structure mini-batch covariance matrices",
    long_description="Use low-rank structure mini-batch covariance matrices"
    + ", in particular gradient covariance and generalized Gauss Newton",
    long_description_content_type="text/markdown",
    url="https://github.com/f-dangel/curvature-noise-spectra",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.6",
)
