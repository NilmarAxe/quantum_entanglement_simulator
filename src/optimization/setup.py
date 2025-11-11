from setuptools import setup, Extension
import pybind11
import os
import sys

# Try to find Eigen3
eigen_paths = [
    '/usr/include/eigen3',
    '/usr/local/include/eigen3',
    '/opt/homebrew/include/eigen3',  # macOS Homebrew
    'C:/Program Files/eigen3',  # Windows
]

eigen_include = None
for path in eigen_paths:
    if os.path.exists(path):
        eigen_include = path
        break

if eigen_include is None:
    print("WARNING: Eigen3 not found in standard locations.")
    print("C++ acceleration will not be available.")
    print("\nTo install Eigen3:")
    print("  Ubuntu/Debian: sudo apt-get install libeigen3-dev")
    print("  macOS: brew install eigen")
    print("  Windows: Download from https://eigen.tuxfamily.org/")
    
    # Create dummy module that raises ImportError
    with open('matrix_ops_cpp_dummy.py', 'w') as f:
        f.write('''
def fast_matrix_multiply(*args, **kwargs):
    raise ImportError("C++ acceleration not available - Eigen3 not found during installation")

def fast_eigenvalues(*args, **kwargs):
    raise ImportError("C++ acceleration not available - Eigen3 not found during installation")

def tensor_product(*args, **kwargs):
    raise ImportError("C++ acceleration not available - Eigen3 not found during installation")
''')
    sys.exit(0)

include_dirs = [
    pybind11.get_include(),
    pybind11.get_include(user=True),
    eigen_include,
]

ext_modules = [
    Extension(
        'matrix_ops_cpp',
        ['matrix_ops.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++14', '-O3', '-march=native'] if sys.platform != 'win32' 
                          else ['/std:c++14', '/O2'],
        extra_link_args=['-fopenmp'] if sys.platform != 'win32' else [],
    ),
]

setup(
    name='matrix_ops_cpp',
    version='1.1',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.11.1'],
)