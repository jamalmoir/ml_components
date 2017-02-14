from distutils.core import setup

setup(
    name='ML Components',
    version='0.1.0',
    author='Jamal Moir',
    author_email='jamal@jamalmoir.com',
    packages=['ml_components', 'mlcomponents.models'],
    license='LICENSE.txt',
    description='Useful towel-related stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.12.0",
        "pandas == 0.19.2",
    ],
)