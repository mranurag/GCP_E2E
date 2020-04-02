from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'scikit-learn==0.22.1',
  'pandas==0.25.0',
  'numpy',
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)