from distutils.core import setup

setup(
    name='desr',
    version='0.1',
    packages=['desr',],
    author='Richard Tanburn',
    author_email='richard.tanburn@gmail.com',
    description='Simplify ordinary differential equations by finding scaling symmetries.',
    license='Apache License Version 2.0',
    long_description=open('README.md').read(),
)