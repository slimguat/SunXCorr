#!/usr/bin/env python
"""Setup script for the Solar Image Coalignment library."""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "Solar image coalignment library for SPICE/FSI alignment"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.20',
        'sunpy>=5.0',
        'astropy>=5.0',
        'matplotlib>=3.5',
        'scipy>=1.7',
        'reproject>=0.10',
    ]

setup(
    name='solar-coalignment',
    version='1.0.0',
    description='High-precision solar image coalignment library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Salim Zergua',
    author_email='salim.zergua@ias.u-psud.fr',
    url='https://github.com/smzergua/solar-coalignment',
    py_modules=[
        'coaligner',
        'coalign_debug',
        'coalign_helpers',
        'coalign_workers',
        'coalign_preprocess',
        'help_funcs',
        'slimfunc_correlation_effort',
    ],
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
        ],
        'acceleration': [
            'numba>=0.56',
        ],
    },
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='solar physics image alignment coalignment SPICE EUI cross-correlation',
    project_urls={
        'Source': 'https://github.com/smzergua/solar-coalignment',
    },
    include_package_data=True,
    zip_safe=False,
)
