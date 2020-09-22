
#2020-03-12 00:00:00
#2020-03-12 00:03:00

test = ['2020-03-12 00:00:00',
        '2020-03-12 00:03:00',
        '2020-03-12 00:00:30',
        '2020-03-12 00:03:30',
        '2020-03-12 00:17:32'
        ]
def time2stamp(str):
    s = int(str[-2:])
    m = int(str[-5:-3])
    h = int(str[-8:-6])
    stamp = h*60+m
    return  stamp


if __name__ == '__main__':


    for item in test:
        stamp = time2stamp(item)
        print(stamp)