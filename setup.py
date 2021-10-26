"""Setup file for vivit."""

from setuptools import find_packages, setup

setup(
    author="Felix Dangel, Lukas Tatzel",
    name="vivit",
    version="0.0.1",
    description="Access curvature through the GGN's low-rank structure with BackPACK",
    long_description_content_type="text/markdown",
    url="https://github.com/f-dangel/vivit",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
)
