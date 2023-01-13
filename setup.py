from setuptools import setup, find_packages

setup(
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
)
