import os
from setuptools import setup, find_packages


os.system('python update_data.py')

setup(
    name='my-project',
    version='1.0',
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)