import setuptools

# with open("README.md", "r", encoding = "utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name = "aang",
    version = "0.2.868",
    author = "seba3y",
    author_email = "sebaeymohamed43@gmail.com",
    description = "speech translation",
  
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": 'aang'},
    
    packages = setuptools.find_packages(where="aang"),
    install_requires=["torch>=1.13", "torchaudio", "torchvision", "torchtext", 
                      "pytorch-lightning>=1.9", "tokenizers", "pyarabic"],
    python_requires = ">=3.7"
)
