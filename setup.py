import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='laruco',
    version='0.0.1',
    description='The python package for a lazy work with ArUco.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/vlavrik/lazy-aruco',
    author='Dr. Vladimir Lavrik',
    author_email='lavrikvladimir@gmail.com',
    license='',
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires='>=3.7')
