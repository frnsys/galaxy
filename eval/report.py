import os
import jinja2

templateLoader = jinja2.FileSystemLoader(searchpath='eval/tmpl')
templateEnv = jinja2.Environment(loader=templateLoader)

def build_report(filename, data, template=None):
    savepath = 'eval/reports/{0}'.format(filename)

    if template is not None:
        tmpl = templateEnv.get_template(template)
        data = tmpl.render(data)
        savepath += '.html'
    else:
        savepath += '.txt'

    with open(savepath, 'w') as report:
        report.write(data)
    return os.path.abspath(savepath)
