import os
import jinja2

templateLoader = jinja2.FileSystemLoader(searchpath='tmpl')
templateEnv = jinja2.Environment(loader=templateLoader)

def build_report(filename, results):
    tmpl = templateEnv.get_template('report.html')
    savepath = 'reports/{0}.html'.format(filename)
    html = tmpl.render(results)


    with open(savepath, 'w') as report:
        report.write(html)
    return os.path.abspath(savepath)
