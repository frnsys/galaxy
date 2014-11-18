import logging
import click

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
        return '\n'.join([self.divider, row, self.divider.replace('-', '=')])

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
        return '\n'.join([row, self.divider])

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
