from setuptools import setup, find_packages

setup(
    name='Outlier',
    version='0.1.0',
    url='https://github.com/liuxx479/Outlier.git',
    author='Jia Liu, Vanessa Boehm, Francois Lanusse',
    description='Code for outlier detection in numerical simulations',
    packages=find_packages(),
    package_data={'outlier': ['datasets/url_checksums/*.txt']},
    install_requires=['tensorflow_datasets', 'tensorflow', 'astropy'],
)
