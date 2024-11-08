from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "cgrad.tensor", 
        sources=[
            'cgrad/tensor.pyx',
            'cgrad/tensor_ops/tensor.c',
            'cgrad/gemm/matmulNd.c',
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O2"]
    ),
    Extension(
        "cgrad.optium.basic_ops",
        sources=[
            'cgrad/optium/basic_ops.pyx',
            'cgrad/optium/tensor_ops.c',
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"] 
    ),
    Extension(
        "cgrad.autograd.grad_funcs",
        sources=[
            'cgrad/autograd/grad_funcs.pyx',
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O2"] 
    ),
    Extension(
        "cgrad.autograd.graph",
        sources=[
            'cgrad/autograd/graph.pyx'
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O2"] 
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
    ext_modules=cythonize(ext_modules), 
    license="MIT",  
    install_requires=["numpy","cython"], 
    package_dir={'': '.'},  
    packages=find_packages(), 
    package_data={
        'cgrad.Tensor':['cgrad/tensor.pyi'],
        'cgrad':['cgrad/optium/basic_ops.pyi'],
        'cgrad.Autograd':['cgrad/autograd/grad_funcs.pyi']
    },
    include_package_data=True,  
    zip_safe=False  
)
