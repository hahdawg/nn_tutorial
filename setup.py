from setuptools import setup, find_packages

setup(
    name="nn_tutorial",
    description="",
    author="Andrew Hah",
    author_email="hahdawg@yahoo.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.ini"]
    },
    install_requires=[
        "torch>=1.8.0",
        "numpy",
        "scikit_learn"
    ],
    zip_safe=False
)
