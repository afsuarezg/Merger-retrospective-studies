from setuptools import setup, find_packages


setup(
    name='nielsen_data_cleaning',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    author='Andrés Felipe Suárez G.',
    author_email='asuarezg@stanford.edu',
    description='A package for cleaning Nielsen data',
    url='https://github.com/afsuarezg/Nielsen_data_cleaning',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)