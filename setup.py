import versioneer
from setuptools import find_namespace_packages, setup

short_description = "A project (and object) for storing, manipulating, and converting molecular mechanics data."

with open("README.md") as handle:
    long_description = handle.read()


setup(
    name="openff-interchange",
    author="Open Force Field Initiative",
    author_email="info@openforcefield.org",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    packages=find_namespace_packages(),
    include_package_data=True,
    setup_requires=[],
)
