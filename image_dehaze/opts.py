import argparse

class OPTIONS:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='AMOS init')

        self.parser.add_argument('--dir', type=str, default='/home/roit/datasets/npmcm2020/ew',
                            help='机场视频2对应')
        self.parser.add_argument('--mask_points',default=(350,540,783,720))
        #self.parser.add_argument('--mask_points',default=(450,633,646,720))

    def args(self):
        self.options = self.parser.parse_args()
        return self.options
