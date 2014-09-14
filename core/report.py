import os
import jinja2

templateLoader = jinja2.FileSystemLoader(searchpath='tmpl')
templateEnv = jinja2.Environment(loader=templateLoader)

def build_report(template, filename, data):
    tmpl = templateEnv.get_template(template)
    savepath = 'reports/{0}.html'.format(filename)
    html = tmpl.render(data)

    with open(savepath, 'w') as report:
        report.write(html)
    return os.path.abspath(savepath)
