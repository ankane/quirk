from setuptools import setup

setup(
    name='quirk',
    version='0.1.0',
    description='Your data science sidekick',
    url='https://github.com/ankane/quirk',
    author='Andrew Kane',
    author_email='andrew@chartkick.com',
    license='MIT',
    packages=['quirk'],
    install_requires=[
        'pandas',
        'seaborn',
        'sklearn',
        'matplotlib',
        'scipy'
    ],
    zip_safe=False
)
