from setuptools import setup, find_packages

# Read requirements.txt (skip blanks and full-line comments)
with open('requirements.txt') as f:
    requirements = [
        line.strip()
        for line in f.read().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="mber",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    author="Erik Swanson",
    author_email="erik@manifold.bio",
    description=(
        "Theta Team internal fork of mBER (Manifold Binder Engineering and Refinement). "
        "Upstream: https://github.com/manifoldbio/mber-open (Copyright 2025 Manifold Bio, MIT)."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhaorui-bi/Theta-mBER",
    project_urls={
        "Upstream": "https://github.com/manifoldbio/mber-open",
        "Paper": "https://www.biorxiv.org/content/10.1101/2025.09.26.678877v1",
        "License": "https://github.com/zhaorui-bi/Theta-mBER/blob/main/LICENSE",
        "Third-Party Notices": "https://github.com/zhaorui-bi/Theta-mBER/blob/main/THIRD_PARTY_NOTICES.md",
    },
    license="MIT",
    license_files=(
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "src/mber/models/alphafold/LICENSE",
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
