# Galaxy

These are some implementations of text clustering algorithms, though only the implementation of
Incremental Hierarchical Agglomerative Clustering (IHAC) has been polished and completed.

This project was put together specifically for [Argos](https://github.com/publicscience/argos)
so some of the setup is tailored to that, though I eventually hope to make it a bit more general.

## Setup

There is some prep you need to do for Pytables (the `tables` package):

    pip install numexpr cython

    # On OSX
    brew tap homebrew/science
    brew install hdf5

    # On Linux
    sudo apt-get install libhdf5-dev

There are other additional dependencies which you can install using the provided `setup` script.

Then you can install `galaxy` with either:

    pip install git+git://github.com/ftzeng/galaxy

or clone this repo and then install it from there (useful if you are actively working on this project):

    pip install --editable .

Note that you many need to manually upgrade the `topia.termextract` module to this git fork which is Py3 compatible:

     pip install -U git+git://github.com/BowdoinOrient/topia.termextract.git

Then you should configure things as needed (see the next section) and then train your pipelines:

    $ python run.py train /path/to/training/data.json

This expects that the data is a list of dictionaries, with `title` and `text` keys.
Depending on how much data you have, this could take a long time.

## Config

You can configure a few things by setting them on `galaxy.conf`:

    conf.PIPELINE_PATH = where you pickled pipelines are stored.
    conf.STANFORD      = a dict specifying the host & port of your Stanford NER server.
    conf.SPOTLIGHT     = a dict specifying the host & port of your DBpedia Spotlight server.

You should set these _before_ you load any of the other `galaxy` modules.

## Usage

    # Train the feature pipelines (text and concepts):
    $ python run.py train /path/to/training/data.json

    # Evaluate event clustering (on labeled data):
    # The `--incremental` flag will break the input into chunks of random sizes.
    $ python run.py eval /path/to/eval/data.json <approach> [--incremental]

    # Run clustering on unlabeled data:
    $ python run.py cluster /path/to/test/data.json

    # Fix up encoding errors in data:
    $ python run.py clean /path/to/data.json

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
