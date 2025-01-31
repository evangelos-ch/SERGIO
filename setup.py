import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = "1.9.3"
PACKAGE_NAME = "sergio-scSim"
AUTHOR = "Payam Dibaeinia"
AUTHOR_EMAIL = "dibaein2@illinois.edu"
URL = "https://github.com/PayamDiba/SERGIO"

LICENSE = "GNU GENERAL PUBLIC LICENSE"
DESCRIPTION = "SERGIO is a simulator for single-cell expression data guided by gene regulatory networks."
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ["numpy", "pandas", "networkx", "cma", "matplotlib", "scikit-learn"]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    package_data={"SERGIO": ["py.typed", "*.txt", "*.node", "*.source"]},
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
