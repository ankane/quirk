from __future__ import print_function


class Terminal(object):
    @staticmethod
    def header(text):
        print('# %s\n' % text)

    @staticmethod
    def subheader(text):
        print('## %s\n' % text)

    @staticmethod
    def paragraph(*lines):
        for line in lines:
            print(line)
        print('')

    @staticmethod
    def plot(_):
        pass

    @staticmethod
    def table(table):
        print(table)
        print('')
