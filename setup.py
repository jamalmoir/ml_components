from distutils.core import setup

setup(
    name='ML Components',
    version='0.1.0',
    author='Jamal Moir',
    author_email='jamal@jamalmoir.com',
    packages=['ml_components', 'ml_components.models', 'ml_components.models.utils'],
    license='LICENSE.txt',
    description='A Library of Maching Learning Models and Algorithms.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.12.0",
        "pandas == 0.19.2",
    ],
)