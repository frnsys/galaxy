from difflib import SequenceMatcher

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


def process(existing, new):
    """
    Args:

        existing => {event_id => [article_ids], ...}
        new => [ [article_ids], ... ]

    Returns which _existing_ clusters have been _updated_,
    which ones should be _created_,
    and which ones should be _deleted_.
    """
    to_update = {}
    to_delete = []

    # Keep sorting consistent.
    candidates = [sorted(clus) for clus in new_clusters]

    # For each existing cluster,
    for id, clus in event_map.items():
        # Keep sorting consistent.
        clus = sorted(clus)
        s = SequenceMatcher(a=clus)

        # Compare to each remaining candidate cluster...
        for i, new_clus in enumerate(candidates):
            s.set_seq2(new_clus)

            # If the similiarty is over 50%, consider the new
            # cluster the same as the old cluster.
            if s.ratio() >= 0.5:
                # This new cluster is now claimed,
                # remove it from the candidates.
                to_update[id] = candidates.pop(i)
                break

        # If there were no matches for the old cluster,
        # delete it.
        else:
            to_delete.append(id)

    # Any remaining candidates are considered new, independent clusters.
    to_create = candidates

    print('Update these clusters: {0}'.format(to_update))
    print('Create these clusters: {0}'.format(to_create))
    print('Delete these clusters: {0}'.format(to_delete))
    return to_update, to_create, to_delete
