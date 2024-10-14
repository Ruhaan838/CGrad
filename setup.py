from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

# Define the extension module
ex = [
    Extension(
        "cgrad.tensor.Tensorwrapper",
        sources=['cgrad/tensor/Tensorwrapper.pyx', 'cgrad/tensor/tensor.c'],
        extra_compile_args=['-arch', 'arm64'],
        extra_link_args=['-arch', 'arm64']
    )
]

setup(
    name="CGrad",
    version="0.1",
    ext_modules=cythonize(ex, annotate=True),
    license="mit",
    install_requires=["numpy"],
    package_dir={'cgrad': 'cgrad'},
    packages=find_packages(where='CGrad'),
)