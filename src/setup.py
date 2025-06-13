import setuptools

setuptools.setup(
    name='Cellexum',
    version='2.0.0-pre3',
    author='Nicholas Hansen',
    author_email='nicholas.2000@live.dk',
    description='A program designed for image analysis of cells on 2D surfaces.',
    license='MIT',
    install_requires=['matplotlib', 'pandas', 'opencv-python, python-javabridge', 'numpy', 'python-bioformats'],
)
