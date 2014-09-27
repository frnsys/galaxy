You can use the env you set up for the core argos project.

## Usage

    # Train the feature pipeline:
    $ python run.py train /path/to/training/data.json

    # Evaluate event clustering (on labeled data):
    $ python run.py evaluate /path/to/evaluation/data.json

    # Run clustering on unlabeled data:
    $ python run.py test /path/to/test/data.json