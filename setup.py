from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "cgrad.tensor.Tensorwrapper", 
        sources=[
            'cgrad/tensor/Tensorwrapper.pyx',
            'cgrad/tensor/tensor.c',
            'cgrad/gemm/matmulNd.c',
        ],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "cgrad.randoms.random_methods",
        sources=[
            'cgrad/randoms/random_methods.pyx',
            'cgrad/randoms/random_tensor.c'
        ],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "cgrad.autograd.grad_funcs",
        sources=[
            'cgrad/autograd/grad_funcs.pyx',
        ],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "cgrad.autograd.graph",
        sources=[
            'cgrad/autograd/graph.pyx'
        ],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="cgrad", 
    version="0.0.3", 
    description="A Cython-based tensor and autograd library",  
    long_description=open('README.md', encoding='utf-8').read(),  
    long_description_content_type='text/markdown',  
    author="Ruhaan",  
    author_email="ruhaan123dalal@gmail.com", 
    url="https://github.com/Ruhaan838/CGrad", 
    ext_modules=cythonize(ext_modules, annotate=True), 
    license="MIT",  
    install_requires=["numpy","cython"], 
    package_dir={'': '.'},  
    packages=find_packages(), 
    package_data={
        'cgrad.Tensor':['cgrad/tensor/Tensorwrapper.pyi'],
        'cgrad':['cgrad/randoms/random_methods.pyi'],
    },
    include_package_data=True,  
    zip_safe=False  
)