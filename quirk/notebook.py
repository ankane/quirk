from IPython.display import display
from IPython.core.display import HTML
import matplotlib.pyplot as plt


class Notebook(object):
    def header(self, text):
        self.html('<h2>%s</h2>' % text)

    def subheader(self, text):
        self.html('<h3>%s</h3>' % text)

    def paragraph(self, *lines):
        self.html('<p>%s</p>' % '<br />'.join(lines))

    @staticmethod
    def plot(_):
        plt.show()

    @staticmethod
    def table(table):
        display(table)

    @staticmethod
    def html(html):
        display(HTML(html))
