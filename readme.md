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
