from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='glm_saga',
    version='0.1.2',
    description="A PyTorch solver for elastic net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Eric Wong',
    author_email='wongeric@mit.edu',
    platforms=['any'],
    license="MIT",
    url='https://github.com/madrylab/glm_saga',
    packages=['glm_saga'],
    install_requires=[
        'torch>=1.0'
    ]
)
