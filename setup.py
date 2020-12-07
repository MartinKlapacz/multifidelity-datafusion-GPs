from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='multifidelityGPs',
    version='0.1.0',
    description='implementation of my bachelor thesis',
    long_description=readme,
    author='Martin Klapacz',
    author_email='klapacz.martin@gmail.com',
    url='https://github.com/MartinKlapacz/MultifidelityGPs.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
)