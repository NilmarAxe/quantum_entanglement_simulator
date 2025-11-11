from setuptools import setup, find_packages, Extension
import pybind11

# C++ extension for matrix operations
cpp_extension = Extension(
    'src.optimization.matrix_ops_cpp',
    sources=['src/optimization/matrix_ops.cpp'],
    include_dirs=[
        pybind11.get_include(),
        pybind11.get_include(user=True),
        '/usr/include/eigen3',
    ],
    language='c++',
    extra_compile_args=['-std=c++14', '-O3', '-march=native', '-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='quantum-entanglement-simulator',
    version='1.0.0',
    description='Advanced quantum entanglement simulator with state control',
    author='Quantum Research Team',
    packages=find_packages(),
    install_requires=[
        'qiskit>=0.45.0',
        'qiskit-aer>=0.13.0',
        'numpy>=1.24.3',
        'scipy>=1.11.3',
        'matplotlib>=3.8.0',
        'seaborn>=0.12.2',
        'pandas>=2.1.1',
        'pybind11>=2.11.1',
    ],
    ext_modules=[cpp_extension],
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'quantum-sim=main:main',
        ],
    },
)