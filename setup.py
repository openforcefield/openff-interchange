import sys
from setuptools import setup, find_namespace_packages
import versioneer

short_description = "A project (and object) for storing, manipulating, and converting molecular mechanics data."

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='openff-interchange',
    author='Open Force Field Initiative',
    author_email='info@openforcefield.org',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    packages=find_namespace_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "LennardJones14 = openff.interchange.handlers.plugins.nonbonded:LennardJones14",
        ]
    },

)
