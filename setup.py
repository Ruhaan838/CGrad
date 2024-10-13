from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os

# Define the extension module
ex = [
    Extension(
        "CGrad.tensor.Tensorwrapper",
        sources=['CGrad/tensor/Tensorwrapper.pyx', 'CGrad/tensor/tensor.c'],
        extra_compile_args=['-arch', 'arm64'],
        extra_link_args=['-arch', 'arm64']
    )
]

setup(
    name="CGrad",
    version="0.1",
    ext_modules=cythonize(ex),
    license="mit",
    install_requires=["numpy"],
    package_dir={'CGrad': 'CGrad'},
    packages=find_packages(where='CGrad'),
    language_level=3,
    options={
        'build_ext': {
            'build_lib': os.path.abspath('CGrad'),  # Define custom output directory
        }
    }
)