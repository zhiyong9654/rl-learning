import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Agents", # Replace with your own username
    version="0.0.1",
    author="dq",
    author_email="dq@example.com",
    description="A set of RL Agents to solve classic RL problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires=">=3.7",
)