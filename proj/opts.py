import argparse

class OPTIONS:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='AMOS init')

        self.parser.add_argument('--airports_csv_in', type=str, default='/home/roit/datasets/npmcm2020/AMOS/AMOS20200313/VIS_R06_12.csv',
                            help='机场视频2对应')
        self.parser.add_argument('--airports_csv_out', type=str, default='/home/roit/datasets/npmcm2020/AMOS/AMOS20200313/airports.csv')

    def args(self):
        self.options = self.parser.parse_args()
        return self.options
