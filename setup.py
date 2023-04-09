import os
import setuptools

from pathlib import Path


def load_requirements():
    with open('requirements.txt') as infile:
        install_requires = [item for item in infile.readlines()
                            if not item.startswith('git+')]

        return install_requires


setuptools.setup(
    name="imageprocess",
    version="0.0.1",
    author="Elliott Polzin",
    author_email="elliott.polzin3@gmail.com",
    description="Image alignment methods",
    url="",
    packages=setuptools.find_packages('src'),
    package_dir={"": "src"},
    package_data={"imageprocess": ["geo/postcodes/gb_postcodes.json",
                                "core/cli/peak_jsprit-0.2.4-SNAPSHOT.jar"]},
    python_requires='>=3.6',
    install_requires=load_requirements()
)