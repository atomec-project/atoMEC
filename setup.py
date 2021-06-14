from setuptools import setup, find_packages


with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="atoMEC",
    version="0.1.0",
    description="KS-DFT average-atom code",
    long_description=readme,
    author="Tim Callow",
    author_email="t.callow@hzdr.de",
    url="https://github.com/atomec-project/atoMEC",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=["numpy", "scipy", "mendeleev", "tabulate", "pylibxc2", "joblib"],
    python_requires=">=3.6",
)
