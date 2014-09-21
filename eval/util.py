import click

def progress(iter, label):
    fill_char = click.style('#', fg='green')
    with click.progressbar(iter, label=label, fill_char=fill_char) as items:
        for item in items:
            yield item
