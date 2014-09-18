import click

def progress(iter, label):
    fill_char = click.style('#', fg='green')
    with click.progressbar(iter, label=label, fill_char=fill_char) as items:
        for item in items:
            yield item


def labels_to_lists(objs, labels):
    """
    Convert a list of objects
    to be a list of lists arranged
    according to a list of labels.
    """
    tmp = {}

    for i, label in enumerate(labels):
        if label not in tmp:
            tmp[label] = []
        tmp[label].append(objs[i])

    return [v for v in tmp.values()]


def changed_clusters(objs, old_labels, new_labels):
    """
    Returns which of the old clusters have changed.
    """
    old = labels_to_lists(objs, old_labels)
    new = labels_to_lists(objs, new_labels)

    for cluster in new:
        if cluster not in old:
            yield cluster
