You can use the env you set up for the core argos project.

## Usage

    # Train the feature pipeline:
    $ python run.py train /path/to/training/data.json

    # Evaluate event clustering (on labeled data):
    $ python run.py evaluate /path/to/evaluation/data.json

    # Run clustering on unlabeled data:
    $ python run.py test /path/to/test/data.json

## Visualization setup
If you want to get visualizations working for the IHAC hierarchy, you have to do a lot
of work (these instructions are for OSX 10.9):

    brew install qt
    brew install graphviz
    pip install -U pyside
    git clone https://github.com/PySide/pyside-setup.git
    cd pyside-setup
    pyside_postinstall.py -install

    # pygraphviz with basic py3 support
    pip install git+git://github.com/ftzeng/pygraphviz.git