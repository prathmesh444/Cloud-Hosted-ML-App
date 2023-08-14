from setuptools import setup, find_packages

def get_requirements():
    total_requirements = []
    with open('requirements.txt') as f:
        module = f.readline()
        module.replace("\n", "")
        if module != "-e .":
            total_requirements.append(module)
    return total_requirements

setup(
    name='MyProject',
    author="Prathmesh Soni",
    author_email="prathmeshsoni6@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    install_requires= get_requirements(),
)