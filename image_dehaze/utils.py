import math

def dis(a,b):

    return math.sqrt(pow(abs(a[0]-b[0]),2)+pow(abs(a[1]-b[1]),2))


p1=(1593,935)
p2=(1817,951)
p3=(1811,1439)
p4=(1357,1337)

d23 =dis(p2,p3)
d34=dis(p3,p4)

#print(d23,d34)


def caculate_intrinsic(xx,yx, xy,yy, xz,yz):

    u=((yx-yy)+(xz-xy)*xx/(yz-yy) - (xz-xx)*xy/(yz-yx) )/( (xz-xy)/(yz-yy)-(xz-xx)/(yz-yx) )

    v = yx+ (xz-xy)*(xx-u)/(yz-yy)
    temp = (xx - u) * (xy - u) + (yx - u)*(xy - v)
    temp=abs(temp)

    f = math.sqrt(temp)

    return u,v,f

if __name__ =='__main__':
# img3 灭点
    xx= 5806.
    yx=4879.

    xy=9627.
    yy=4927.

    xz=8706.
    yz=15988.

    origin_x = 7824
    origin_y= 4490
  # uav 灭点
    xx= 1823.
    yx=1029.

    xy=3345.
    yy=872.

    xz=2765.
    yz=3769.

    origin_x = 2236
    origin_y= 960
    u,v,f = caculate_intrinsic(xx-origin_x, yx-origin_y, xy-origin_x, yy-origin_y, xz-origin_x, yz-origin_y)
    print(u,v,f)
# uav 灭点2
    xx = 1375.
    yx = 1440.

    xy = 3090.
    yy = 1498.

    xz = 2265.
    yz = 3142.

    origin_x = 1580
    origin_y = 1504

    u,v,f = caculate_intrinsic(xx-origin_x, yx-origin_y, xy-origin_x, yy-origin_y, xz-origin_x, yz-origin_y)

    #w_tor = 512/1920
    #h_tor = 256/1080

    #u*=w_tor
    #f*=(w_tor+h_tor)/2
    print(u, v, f)