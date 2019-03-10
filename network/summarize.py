import os


class Summarize:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.markdown_path = os.path.join(self.log_dir, 'summary.md')
        self.markdown = open(self.markdown_path, 'w')

    def make_title(self, heading, heading_level=1):
        self.markdown.write('{} {}  \n'.format('#'*heading_level, heading))

    def make_table(self, data, x_labels=None, y_labels=None):
        table = []
        if x_labels:
            text_row = '| ' + ('| ' if y_labels else '')
            bottom_row = '| ' + ('--- | ' if y_labels else '')
            for label in x_labels:
                text_row += '{} | '.format(label)
                bottom_row += '--- | '
            table.append(text_row)
            table.append(bottom_row)

        for r_index in range(len(data)):
            text_row = '| '
            if y_labels:
                text_row += '**{}** | '.format(y_labels[r_index])
            for c_index in range(len(data[r_index])):
                text_row += '{} | '.format(data[r_index][c_index])
            table.append(text_row)

        table_str = ''
        for text_row in table:
            table_str += text_row + '  \n'
        self.markdown.write(table_str)


if __name__ == '__main__':
    s = Summarize('')
    s.make_title('Test')
    x_labels = ['a', 'b', 'c']
    y_labels = ['x', 'y']
    data = [[1, 2, 3], [4, 5, 6]]
    s.make_table(data, x_labels, y_labels)



