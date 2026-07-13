# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup script for semantic-slam-server package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Semantic SLAM Server"

# Read requirements
def read_requirements():
    """Read requirements from conda environment file."""
    requirements = [
        "torch>=2.0.1",
        "torchvision>=0.15.2", 
        "torchaudio>=2.0.2",
        "numpy>=1.24.3,<1.27.0",
        "opencv-python>=4.9.0,<4.11.0",
        "Pillow>=9.5.0",
        "grpcio>=1.65.0",
        "grpcio-tools>=1.65.0",
        "supervision>=0.14.0",
        "pynvvideocodec>=1.0.0",
        "distinctipy>=1.3.0",
        "open-clip-torch>=2.26.0",
        "transformers>=4.25.1,<4.26.0",
        "openai>=1.37.0",
        "hydra-core>=1.3.0",
        "wandb>=0.17.0",
        "h5py>=3.11.0",
        "tyro>=0.8.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.9.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.10.0,<1.15.0",
    ]
    return requirements

setup(
    name="semantic-slam-server",
    version="1.0.0",
    author="Semantic SLAM Team",
    description="A semantic SLAM server with real-time object detection and mapping",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["slam*", "server*", "config*", "evaluation*"]),
    install_requires=read_requirements(),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "semantic-slam-server=main_server.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "inference_SLAM": ["utilsSLAM/*.yaml"],
    },
)