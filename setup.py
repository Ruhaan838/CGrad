from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "cgrad.tensor.Tensorwrapper", 
        sources=[
            'cgrad/tensor/Tensorwrapper.pyx',  
            'cgrad/tensor/tensor.c',
            'cgrad/gemm/matmulNd.c'
        ],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="CGrad", 
    version="0.1", 
    description="A Cython-based tensor and autograd library",  
    long_description=open('README.md', encoding='utf-8').read(),  
    long_description_content_type='text/markdown',  
    author="Ruhaan",  
    author_email="ruhaan123dalal@gmail.com", 
    url="https://github.com/Ruhaan838/CGrad", 
    classifiers=[  
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: MIT License",
        "Development Status :: 3 - Alpha",  
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    ext_modules=cythonize(ext_modules, annotate=True),  
    license="MIT",  
    install_requires=["numpy"], 
    package_dir={'': '.'},  
    packages=find_packages(), 
    include_package_data=True,  
    zip_safe=False  
)