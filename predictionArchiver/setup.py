from setuptools import setup, find_packages

setup(
    name='predictionArchiver',
    version='0.1.3',
    packages=find_packages(),
    py_modules=['predictionArchiver'],
    install_requires=[
        'pydantic',
        'pandas'
    ],
)