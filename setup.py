from setuptools import setup

import torchmodels


setup(
    name="pytorch-models",
    version=torchmodels.__version__,
    license="MIT",
    description="A thin wrapper for scriptable PyTorch modules",
    author="Kang Min Yoo",
    author_email="kaniblurous@gmail.com",
    url="https://github.com/kaniblu/pytorch-models",
    packages=[
        "torchmodels",
        "torchmodels.modules"
    ],
    scripts=[
        "scripts/scaffold"
    ],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Java",
        "Programming Language :: Python",
    ],
    platforms=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ]
)
