from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='merger_retrospective_studies',
    version='0.1.8',
    packages=find_packages(include=['merger_retrospective_studies', 'merger_retrospective_studies.*']),
    install_requires=requirements,
    author='Andrés Felipe Suárez G.',
    author_email='asuarezg@stanford.edu',
    description='A package for cleaning and processing Nielsen data',
    url='https://github.com/afsuarezg/Nielsen_data_cleaning',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7,<3.12',
)