import setuptools

setuptools.setup(
    name='cellexum',
    version='1.0.1',
    author='Nicholas Hansen',
    author_email='nicholas.2000@live.dk',
    description='A specified pack of software designed to analyze NTSA surfaces.',
    license='MIT',
    install_requires=['nanoscipy', 'pandas', 'tqdm', 'opencv-python'],
)
