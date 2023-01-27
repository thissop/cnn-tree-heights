from distutils.core import setup

requirements  = []

with open('requirements.txt', 'r') as f:
    for line in f: 
        requirements.append(line.replace('\n',''))

setup(
    name='cnnheights',
    version='0.1dev',
    author='Thaddaeus Kiker',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires = requirements
)