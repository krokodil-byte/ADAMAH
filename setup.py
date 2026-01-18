from setuptools import setup, find_packages

setup(
    name="adamah",
    version="4.1.0",
    description="Map-Centric GPU Compute Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Samuele Scuglia",
    license="CC-BY-NC-4.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "adamah": ["*.c", "*.h", "shaders/*.spv", "shaders/*.comp"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy"],
    keywords=["gpu", "vulkan", "compute", "neural-network", "transformer", "cuda-alternative"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Operating System :: POSIX :: Linux",
    ],
    url="https://github.com/krokodil-byte/ADAMAH",
)
