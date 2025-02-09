import setuptools

setuptools.setup(
    name='NTSA Img Processing',
    version='2',
    author='Nicholas Hansen',
    author_email='nicholas.2000@live.dk',
    description='A specified pack of software designed to analyze NTSA surfaces.',
    license='MIT',
    install_requires=['matplotlib', 'pandas', 'tqdm', 'opencv-python, python-javabridge', 'numpy', 'python-bioformats'],
)
