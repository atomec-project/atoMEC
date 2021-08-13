from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="atoMEC",
    version="1.0.0",
    description="KS-DFT average-atom code",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Tim Callow et al.",
    author_email="t.callow@hzdr.de",
    url="https://github.com/atomec-project/atoMEC",
    license=license,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires=">=3.6",
)
