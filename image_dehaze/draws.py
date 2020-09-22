
from opts import OPTIONS
from path import Path
import numpy as np
import cv2
def main(args):
    baseline = np.load('./002.npy')

    dir = Path(args.dir)
    files = dir.files()
    files.sort()
    for item in files:
        pass


if __name__ == '__main__':
    args = OPTIONS().args()

    main(args)
