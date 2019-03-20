import json
import argparse
import os.path as osp
import networkx as nx
import operator
import matplotlib.pyplot as plt


class graph_stats:
    def __init__(self, data):
        """
        Create a graph from a given .json and compute statistics on that.
        :param data: <dict> Loaded database as a dictionary of samples.
        """
        self.G = nx.DiGraph()

        self.missing_family = []
        self.missing_subfamily = []
        self.missing_genus = []
        self.missing_specific_epithet = []

        for sample_id in data:
            sample = data[sample_id]
            if "" not in [sample['family'], sample['subfamily'], sample['genus'], sample['specific_epithet']]:
                self.G.add_edge(sample['family'], sample['subfamily'])
                self.G.add_edge(sample['subfamily'], sample['genus'])
                self.G.add_edge(sample['genus'], sample['specific_epithet'])

                for label_level in ['family', 'subfamily', 'genus', 'specific_epithet']:
                    if 'count' not in self.G.nodes[sample[label_level]]:
                        self.G.nodes[sample[label_level]]['count'] = 0
                        self.G.nodes[sample[label_level]]['level'] = label_level
                    self.G.nodes[sample[label_level]]['count'] += 1

            for label_level in ['family', 'subfamily', 'genus', 'specific_epithet']:
                if sample[label_level] == "":
                    getattr(self, 'missing_{}'.format(label_level)).append(sample_id)

        self.family_nodes = [node_data for node_data in self.G.nodes.data() if node_data[1]['level'] == 'family']
        self.subfamily_nodes = [node_data for node_data in self.G.nodes.data() if node_data[1]['level'] == 'subfamily']
        self.genus_nodes = [node_data for node_data in self.G.nodes.data() if node_data[1]['level'] == 'genus']
        self.specific_epithet_nodes = [node_data for node_data in self.G.nodes.data() if
                                       node_data[1]['level'] == 'specific_epithet']

        nx.draw(self.G)
        plt.show()

        self.in_degree = self.G.in_degree
        self.out_degree = self.G.out_degree

        self.print_stats()

    def get_max_degree(self, level=None):
        """
        Function to compute the max in and out degree for a given hierarchy level, otherwise computes them for the
        complete graph.
        :param level: <str> One of the following: ['family', 'subfamily', 'genus', 'specific_epithet', None]
        :return: <int, int> max_in_degree, max_out_degree
        """
        if level not in ['family', 'subfamily', 'genus', 'specific_epithet', None]:
            raise ValueError('Invalid option {}. Use one of {}'
                             .format(level, ['family', 'subfamily', 'genus', 'specific_epithet', None]))

        if level is None:
            max_in_degree = max(dict(self.in_degree).items(), key=operator.itemgetter(1))
            max_out_degree = max(dict(self.out_degree).items(), key=operator.itemgetter(1))
            print("Max in degree: {}".format(max_in_degree))
            print("Max out degree: {}".format(max_out_degree))
        else:
            d = dict(self.in_degree)
            max_in_degree = max({f[0]: d[f[0]] for f in getattr(self, '{}_nodes'.format(level))}.items(),
                                key=operator.itemgetter(1))
            print("Max in degree for {}: {}".format(level, max_in_degree))

            d = dict(self.out_degree)
            max_out_degree = max({f[0]: d[f[0]] for f in getattr(self, '{}_nodes'.format(level))}.items(),
                                 key=operator.itemgetter(1))
            print("Max out degree for {}: {}".format(level, max_out_degree))

        return max_in_degree, max_out_degree

    def print_stats(self):
        """
        Print stats for the loaded dataset.
        :return: None
        """
        print("Number of edges: {}".format(self.G.size()))
        print("Number of nodes: {}".format(len(self.G)))

        print("Number of families: {}".format(len(self.family_nodes)))
        print("Number of subfamilies: {}".format(len(self.subfamily_nodes)))
        print("Number of genera: {}".format(len(self.genus_nodes)))
        print("Number of specific epithets: {}".format(len(self.specific_epithet_nodes)))

        for label_level in ['family', 'subfamily', 'genus', 'specific_epithet']:
            max_val = None
            for node in getattr(self, '{}_nodes'.format(label_level)):
                if max_val is None:
                    max_val = node
                if max_val[1]['count'] < node[1]['count']:
                    max_val = node
            print("Maximum specimens belong to the {} {}".format(label_level, max_val))

        self.get_max_degree(level='family')
        self.get_max_degree(level='subfamily')
        self.get_max_degree(level='genus')
        self.get_max_degree(level='specific_epithet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", help='Use the mini database for testing/debugging.', action='store_true')
    args = parser.parse_args()

    infile = 'database'
    if args.mini:
        infile = 'mini_database'

    if osp.isfile('../database/{}.json'.format(infile)):
        with open('../database/{}.json'.format(infile)) as json_file:
            data = json.load(json_file)
    else:
        print("File does not exist!")
        exit()

    gs = graph_stats(data)
