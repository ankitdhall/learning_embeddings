import json
import argparse
import os.path as osp


class graph_stats:
    def __init__(self, data):
        self.dag = {}
        self.f_counter = {}
        self.sub_f_counter = {}
        self.g_counter = {}
        self.se_counter = {}

        self.missing_family = []
        self.missing_subfamily = []
        self.missing_genus = []
        self.missing_specific_epithet = []

        for sample_id in data:
            sample = data[sample_id]
            if "" not in [sample['family'], sample['subfamily'], sample['genus'], sample['specific_epithet']]:
                self.add(sample['family'], sample['subfamily'], sample['genus'], sample['specific_epithet'])
            for label_level in ['family', 'subfamily', 'genus', 'specific_epithet']:
                if sample[label_level] == "":
                    getattr(self, 'missing_{}'.format(label_level)).append(sample_id)

        self.print_stats()
        self.find_largest_degrees()

    def add(self, family, subfamily, genus, specific_epithet):
        if family is not None:
            if family not in self.dag:
                self.f_counter[family] = 1
                self.dag[family] = {}
            else:
                self.f_counter[family] += 1
        if subfamily is not None:
            if subfamily not in self.dag[family]:
                self.sub_f_counter[subfamily] = 1
                self.dag[family][subfamily] = {}
            else:
                self.sub_f_counter[subfamily] += 1
        if genus is not None:
            if genus not in self.dag[family][subfamily]:
                self.g_counter[genus] = 1
                self.dag[family][subfamily][genus] = {}
            else:
                self.g_counter[genus] += 1
        if specific_epithet is not None:
            if specific_epithet not in self.dag[family][subfamily][genus]:
                self.se_counter[specific_epithet] = 1
                self.dag[family][subfamily][genus][specific_epithet] = {}
            else:
                self.se_counter[specific_epithet] += 1
        # print(self.dag)
        # print(self.f_counter)
        # print(self.sub_f_counter)
        # print(self.g_counter)
        # print(self.se_counter)

    def find_largest_degrees(self):
        max_family = max([len(self.dag[family])
                          for family in self.dag])

        max_subfamily = max([len(self.dag[family][subfamily])
                             for family in self.dag
                             for subfamily in self.dag[family]])

        max_genus = max([len(self.dag[family][subfamily][genus])
                         for family in self.dag
                         for subfamily in self.dag[family]
                         for genus in self.dag[family][subfamily]])

        print("Max_family: {}, Max_subfamily: {}, Max_genus: {}".format(max_family, max_subfamily, max_genus))
        return max_family, max_subfamily, max_genus



    def print_stats(self):
        # n_family = len(self.dag)
        # family_keys = [key for key in self.dag]
        print("Families: {}".format(sorted(list(self.f_counter.keys()))))

        # n_subfamily = sum([len(self.dag[family]) for family in self.dag])
        # subfamily_keys = [subfamily for family in self.dag for subfamily in self.dag[family]]
        print("Sub-families: {}".format(sorted(list(self.sub_f_counter))))

        # n_genus = sum([len(self.dag[family][subfamily]) for family in self.dag for subfamily in self.dag[family]])
        # genus_keys = [genus
        #               for family in self.dag
        #               for subfamily in self.dag[family]
        #               for genus in self.dag[family][subfamily]]
        print("Genus: {}".format(sorted(list(self.g_counter.keys()))))

        # n_specific_epithet = sum([len(self.dag[family][subfamily][genus])
        #                           for family in self.dag
        #                           for subfamily in self.dag[family]
        #                           for genus in self.dag[family][subfamily]])
        # se_keys = [specific_epithet
        #            for family in self.dag
        #            for subfamily in self.dag[family]
        #            for genus in self.dag[family][subfamily]
        #            for specific_epithet in self.dag[family][subfamily][genus]]
        print("Specific epithet: {}".format(sorted(self.se_counter.keys())))

        # total_nodes = n_family + n_subfamily + n_genus + n_specific_epithet

        unique_counter_entries = len(self.f_counter) + len(self.sub_f_counter) + len(self.g_counter) + len(
            self.se_counter)

        print("Family counts: {}".format(self.f_counter))
        print("Sub-family counts: {}".format(self.sub_f_counter))
        print("Genus counts: {}".format(self.g_counter))
        print("Specific Epithet counts: {}".format(self.se_counter))

        n_samples = len(data)
        print("Dataset has {} samples.".format(n_samples))

        print("Unique number of families: {}".format(len(self.f_counter)))
        print("Unique number of sub-families: {}".format(len(self.sub_f_counter)))
        print("Unique number of genus: {}".format(len(self.g_counter)))
        print("Unique number of specific epithet: {}".format(len(self.se_counter)))

        # print("Total number of nodes: {}".format(total_nodes))
        print("Total number of uniques in counter objects: {}".format(unique_counter_entries))

        print("Missing entries for fields: family: {} subfamily: {} genus: {} se: {}".format(
            len(self.missing_family), len(self.missing_subfamily),
            len(self.missing_genus), len(self.missing_specific_epithet)))


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
