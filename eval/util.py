import sys
import logging
import click

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

# Displays progress for an iterator.
def progress(iter, label):
    fill_char = click.style('#', fg='green')
    with click.progressbar(iter, label=label, fill_char=fill_char) as items:
        for item in items:
            yield item


# Returns a logger which _only_ outputs to a file.
def file_logger(name):
    log_path = 'eval/logs/{0}.log'.format(name)
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# Generates ASCII tables for dictionaries.
class TableGenerator():
    def __init__(self, keys):
        self.keys = keys
        self.base_padding = 4
        self.min_col_width = 20

    def build_headers(self):
        self.divider = ''
        row = ''
        for key in self.keys:
            row += self._build_column(key)
            self.divider += self._build_column(len(key) * '-', pad_char='-')
        return '\n'.join([row, self.divider])

    def build_row(self, data):
        row = ''
        for key in self.keys:
            # Convert value to string.
            val = str(data[key])

            # Calculate necessary padding.
            col_width = self.base_padding + len(key)
            if col_width < self.min_col_width:
                col_width = self.min_col_width
            p = col_width - len(val)

            if p < 0:
                val = val[:p]
                p = 0
            row += self._build_column(val, padding=p)
        return row

    def _build_column(self, text, pad_char=' ', padding=None):
        # Fallback to base padding if none is specified.
        if padding is None:
            padding = self.base_padding

        # Expand padding if needed to meet the minimum column width.
        if padding + len(text) < self.min_col_width:
            padding = self.min_col_width - len(text)

        # Calculate left and right padding.
        pad_l = int(round(padding/2.))
        pad_r = int(padding-pad_l)

        return '|' + (pad_l * pad_char) + text + (pad_r * pad_char) + '|'


def progress_bar(percent, elapsed_time):
    """
    Show a progress bar.
    """
    if percent == 0:
        estimated = 0
    else:
        estimated = elapsed_time/percent
    remaining = estimated - elapsed_time
    percent *= 100

    if remaining > 3600:
        countdown = '{:8.2f}hrs'.format(remaining/3600)
    elif remaining > 60:
        countdown = '{:8.2f}min'.format(remaining/60)
    else:
        countdown = '{:8.2f}sec'.format(remaining)

    width = 100
    info = '{0:8.3f}% {1}'.format(percent, countdown)
    sys.stdout.write('[{0}] {1}'.format(' ' * width, info))
    sys.stdout.flush()
    sys.stdout.write('\b' * (width+len(info)+2))

    for i in range(int(percent)):
        sys.stdout.write('=')
        sys.stdout.flush()
    sys.stdout.write('\b' * (width+len(info)+2))

    if percent == 100:
        print('\n')
