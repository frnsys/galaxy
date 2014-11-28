from setuptools import setup, find_packages

setup(
    name='galaxy',
    version='0.1.0',
    description='clustering module',
    url='https://github.com/ftzeng/galaxy',
    author='Francis Tseng',
    author_email='f@frnsys.com',
    license='AGPLv3',

    package_dir={'':'galaxy'},
    packages=find_packages('galaxy'),
    dependency_links=[
        'git+git://github.com/dat/pyner',
        'git+git://github.com:ftzeng/topia.termextract'
    ],
    install_requires=[
        'cython',
        'numexpr',
        'ftfy',
        'nltk',
        'numpy',
        'scipy',
        'networkx',
        'scikit-learn',
        'ner',
        'pytz',
        'python-dateutil',
        'jinja2',
        'click',
        'topia.termextract',
        'tables'
    ],
)
