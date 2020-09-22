import numpy as np
import cv2
import matplotlib.pyplot as plt
#src
def getH():
    src_bl = (337, 650)#(337, 650)
    src_br = (708, 650)
    src_tr = (590, 540)
    src_lr = (327, 540)#(327, 540)
    src = np.array([src_bl, src_br, src_tr, src_lr])
    width = 110
    height=160
    # dst
    dst_bl = (0, 0)
    dst_br = (width, 0)
    dst_tr = (width, height)
    dst_lr = (0, height)
    dst = np.array([dst_bl, dst_br, dst_tr, dst_lr])

    (H, status) = cv2.findHomography(src, dst)

    print(H)
    return H


def main():
    H = getH()*10
    h = 330
    w = 480
    src = cv2.imread('./1.bmp')



    dst = cv2.warpPerspective(src=src, M=H, dsize = (w, h))
    dst = cv2.flip(src=dst,flipCode=0)

    plt.imsave('out.jpg', dst)
    pass


if __name__ == '__main__':
    main()


