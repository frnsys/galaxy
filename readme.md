You can use the env you set up for the core argos project.

## Usage

    # Train the feature pipelines (text and concepts):
    $ python run.py train /path/to/training/data.json

    # Evaluate event clustering (on labeled data):
    $ python run.py evaluate /path/to/eval/data.json <approach> [--incremental|--random]

    # Run clustering on unlabeled data:
    $ python run.py cluster /path/to/test/data.json

The `--random` flag will randomize the ordering of the input.
The `--incremental` flag will break the input into chunks of random sizes.


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

For Ubuntu:

    sudo apt-get install graphviz libgraphviz-dev
    sudo apt-get build-dep python-matplotlib
    pip install matplotlib
    sudo apt-get install cmake qt-sdk
    pip install -U pyside

    # pygraphviz with basic py3 support
    pip install git+git://github.com/ftzeng/pygraphviz.git


## Performance

### Speed
Using the `line_profiler` package.

To measure speed, add the `@profile` decorator to any function you want to measure.
Then run then script:

    $ kernprof -lv run.py <your args>

### Memory
Using the `memory_profiler` and `psutil` packages.

To measure memory usage, add the `@profile` decorator to any function you want to measure.
Then run the script:

    $ python -m memory_profiler run.py <your args>
