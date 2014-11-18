from setuptools import setup

setup(
    name='argos.cluster',
    version='0.1.0',
    description='argos clustering module',
    url='https://github.com/publicscience/argos.cluster',
    author='Francis Tseng',
    author_email='f@frnsys.com',
    license='AGPLv3',

    packages=['core'],
    dependency_links=[
        'git+git://github.com/dat/pyner'
    ],
    install_requires=[
        'nltk',
        'numpy',
        'scipy',
        'pytz',
        'python-dateutil',
        'scikit-learn',
        'ner',
        'jinja2',
        'click'
    ],
)
