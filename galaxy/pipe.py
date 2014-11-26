import os
import pickle

from . import conf

def pipeline_path(pipetype):
    return os.path.expanduser(os.path.join(conf.PIPELINE_PATH, '{0}_pipeline.pickle'.format(pipetype)))

def load_pipeline(pipetype):
    pipe_path = pipeline_path(pipetype)

    if not os.path.isfile(pipe_path):
        raise Exception('No pipeline found at {0}. Have you trained one yet?'.format(pipe_path))

    with open(pipe_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def save_pipeline(pipeline, pipetype):
    pipe_path = pipeline_path(pipetype)

    print('Saving pipeline to {0}'.format(pipe_path))
    with open(pipe_path, 'wb') as f:
        pickle.dump(pipeline, f)
