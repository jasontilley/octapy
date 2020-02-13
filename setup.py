import setuptools
import numpy as np
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


with open("README.md", "r") as fh:
    long_description = fh.read()

#sources = ['octapy/data.pyx', 'octapy/data.cpp']

extensions = [Extension("octapy.*", ["octapy/data.pyx"],
                        language='c++', include_dirs=[np.get_include()]),
              Extension("octapy.*", ["octapy/interp_idw.pyx"],
                        language='c++', include_dirs=[np.get_include()])]

setuptools.setup(
    name='octapy',
    version="0.0.1",
    author="Jason Tilley",
    author_email="jason.tilley@usm.edu",
    description="Ocean Connectivity and Tracking Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasontilley/octapy",
    packages=setuptools.find_packages(),
    package_data={"octapy": ["*.pxd"]},
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions, compiler_directives={'language_level': '3'}),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    zip_safe=False
)
