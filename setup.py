from setuptools import setup, find_packages

total_requirements = []
with open('requirements.txt') as f:
    module = f.read().splitlines()
    if module is not "":
        total_requirements.append(module)

setup(
    name='MyProject',
    author="Prathmesh Soni",
    author_email="prathmeshsoni6@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    install_requires= total_requirements,
)