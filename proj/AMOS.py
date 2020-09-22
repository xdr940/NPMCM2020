import pandas as pd
from opts import OPTIONS
from path import Path
from utils import time2stamp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter   ### 今天的主角
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



def main(args):
    pass


def AsMinutes():
    airports_csv_in = Path(args.airports_csv_in)
    df=pd.read_csv(airports_csv_in,delimiter='\t')
    df['timestamp'] = df['CREATEDATE'].apply(lambda c:time2stamp(c))
    itelist=[]
    for idx,row in df.iterrows():
        if row['timestamp'] not in itelist:
            itelist.append(row['timestamp'])
        else:
            df['timestamp'][idx]=-1

    df = df.query('timestamp!=-1')


    df.to_csv(args.airports_csv_out,index=False,sep='\t')

def pltcolumns3d():
    plt_columns = [
        'RVR DATA',
        'RVR_1A',

        'MOR_RAW',
        'MOR_1A',

        'VIS_RAW',
        'VIS1K',
        'VIS1A',
        # 'BL1A' ,
        #'timestamp'
    ]
    airports_csv_in = Path(args.airports_csv_out)
    df = pd.read_csv(airports_csv_in, delimiter='\t')


    x = df['timestamp']
    y = np.ones_like(x.to_numpy())
    fig = plt.figure()
    ax = Axes3D(fig)

    for idx,item in enumerate(plt_columns):
        z=df[item]
        ax.plot(x,y*idx,z)

    plt.legend(plt_columns)

    #for idx,item in enumerate(plt_columns):
     #   plt.plot(df[item]+idx*1000)


    #plt.legend()
    plt.show()
    print('ok')
def pltcolumns2d():
    plt_columns = [
        'RVR DATA',
        'RVR_1A',

        'MOR_RAW',
        'MOR_1A',

        'VIS_RAW',
        'VIS1K',
        'VIS1A',
        # 'BL1A' ,
        #'timestamp'
    ]
    airports_csv_in = Path(args.airports_csv_out)
    df = pd.read_csv(airports_csv_in, delimiter='\t')


    x = df['timestamp']
    fig = plt.figure()

    for idx,item in enumerate(plt_columns):
        z=df[item]
        plt.plot(z)
    plt.legend(plt_columns)

    #for idx,item in enumerate(plt_columns):
     #   plt.plot(df[item]+idx*1000)


    #plt.legend()
    plt.show()
    print('ok')

def corr():
    value_columns=[
    'RVR DATA',
        'RVR_1A',

        'MOR_RAW',
        'MOR_1A',

        'VIS_RAW',
        'VIS1K',
        'VIS1A',
        'timestamp'
    ]
    df = pd.read_csv('airports.csv', delimiter='\t')
    df = df[value_columns]/1000

    def formatnum(x, pos):
        return '$%.1f$x$10^{4}$' % (x / 10000)
    formatter = FuncFormatter(formatnum)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter))  # 格式化y轴刻度
    sns.pairplot(df)
    plt.savefig('./corr.png',dpi=100,pad_inches=0.1,optimize=True)
    #plt.show()

    pass

if __name__ == '__main__':
    args = OPTIONS().args()
    #main(args)
    AsMinutes()
    corr()
