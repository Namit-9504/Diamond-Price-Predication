from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def getting_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n",'') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
        return requirements

setup(
    name="Diamond Price Predication",
    version='0.0.1',
    author='Namit',
    author_email='namitchaudhari9504@gmail.com',
    install_requires=getting_requirements('requirements.txt'),
    packages=find_packages()

)