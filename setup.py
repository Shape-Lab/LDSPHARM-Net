from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r") as fh:
    long_description = fh.read()


ext_modules = []
ext_modules.append(
    CUDAExtension(
        "TriangleSearchCUDA",
        sources=["ldspharmnet/csrc/triangle_search.cpp", "ldspharmnet/csrc/triangle_search_kernel.cu"],
    )
)

setup(
    name="ldspharmnet",
    version="0.2.0",
    license="Apache 2.0",
    author="Seunghwan Lee",
    author_email="shwan@postech.ac.kr",
    description="Isotropic spherical region enlargement and background contraction for sulcal labeling",
    url="https://github.com/Shape-Lab/LDSPHARM-Net",
    keywords=["Isotropic deformation", "Posterior medial cortex", "Spherical CNN", "Sulcal labeling"],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy==2.2.6",
        "scipy==1.15.3",
        "joblib>=0.14.1",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
