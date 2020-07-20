import os,sys,shutil
import cv2
import numpy as np
import imutils
import copy
import configparser
import platform
import time

import warnings
warnings.filterwarnings('ignore')

### global setting
img_dir = './img_yellow/'
hsv_low = np.array([90,60,120]) # 90,160,150
hsv_high = np.array([105,250,250]) # 100,250,240
is_show = True
is_save = False
is_data = True # command line show 4-cornel axis (x,y) data

### os check
os_name=platform.system()
if os_name == 'Windows':
    save_dir = './output/'
elif os_name == 'Linux':
    save_dir = '/dev/shm/measure/'
else:
    print('unknown os: ' + os_name)
    print('this version support debian 9+ and ubuntu 18+ and windows 7+ OS')
    sys.exit(1)

### -------------------------------------------------------------
def im_draw(title,im):
    if not is_show:
        return

    h = im.shape[0]
    w = im.shape[1]
    if w>1280 or h >960:
        scalesize = min(w/1280,h/960)
        w /= scalesize
        h /= scalesize
        w = int(w)
        h = int(h)
    im = cv2.resize(im,(w,h))
    cv2.imshow(title, im)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    if key == 27:
        sys.exit(2)

# pts = [4,1,2]
def im_draw_rect(im,pts):
    x1 = int(pts[0][0][0])
    y1 = int(pts[0][0][1])
    x2 = int(pts[2][0][0])
    y2 = int(pts[2][0][1])


    im = cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
    im_draw('draw rect',im)


def im_draw_line(im,k,b):
    x0 = 0
    x1 = im.shape[1]
    y0 = int(k*x0 +b)
    y1 = int(k*x1 +b)

    im = cv2.line(im,(x0,y0),(x1,y1),(0,255,0),3)

    im_draw('draw line',im)

def im_draw_pt(im,pts):
    lens = len(pts[0])
    for i in range(0,lens):
        x = pts[0][i]
        y = pts[1][i]
        im = cv2.circle(im,(x,y),3,(0,255,255),1)


### ==========================================================
def enhance_colour(im):
    # rgb:
    #lower_rgb = np.array([60, 150, 180])
    #upper_rgb = np.array([150, 210, 220])
    #mask = cv2.inRange(im,lower_rgb,upper_rgb)

    # hsv: 180,255,255 (gimp:360,100,100)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    return mask


def enhance_img(im):
    #im_draw('origin',im)
    im = cv2.GaussianBlur(im, (9, 9), 0)
    im_gray = enhance_colour(im)
    im_gray = cv2.medianBlur(im_gray, 7, 0)
    im_gray = im_gray.astype(np.uint8)

    #im_draw('enhance_img',im_gray)
    return im_gray



def approx_fit_quadrilateral(contour):
    # fit quadrilateral
    epsilon = 0.1*cv2.arcLength(contour,True)
    approx_points = cv2.approxPolyDP(contour,epsilon,True)
    return approx_points



def get_max_contour(im_th):
    # get max contour
    contours = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours)==0:
        return None
    else:
        max_c = max(contours, key=cv2.contourArea)
        return max_c


def get_max_threshold_img(im_gray,l1,l2): # 100,150
    _,thres = cv2.threshold(im_gray,l1,255,cv2.THRESH_BINARY)
    thres = np.where(thres<l2, 0, thres)

    #im_draw('thres',thres)

    # get max contour
    c = get_max_contour(thres)
    if c is None:
        return None,None

    # apprex fit quadrilateral
    approx_pt = approx_fit_quadrilateral(c)

    # remake threshold image, with max contour and fill 255
    th_new = np.zeros((thres.shape[0],thres.shape[1]),dtype=np.uint8)
    th_new = cv2.fillPoly(th_new, pts=[approx_pt], color=(255))
    thres = cv2.bitwise_or(th_new,thres)

    #im_draw('max c in image',thres)

    return thres,c

def get_outer_rect(pt,pt0):
    ptd = copy.deepcopy(pt)

    pt1 = (pt0+1) % 4
    pt2 = (pt1+1) % 4
    pt3 = (pt2+1) % 4

    gap = 30
    ngap = 20

    #print('pt info = {}'.format(len(pt)))

    # pt0 to pt3 line check
    ck = [ pt[pt0][0][0] - pt[pt3][0][0],
           pt[pt0][0][1] - pt[pt3][0][1] ]


    # make new pt1 and pt2
    if abs(ck[0]) > abs(ck[1]): # x move > y move, move y
        if pt[pt0][0][1] < pt[pt1][0][1] and \
           pt[pt3][0][1] < pt[pt2][0][1]:
            gap = (-1)*gap
            ngap = (-1)*ngap
        ptd[pt1][0] = [ pt[pt0][0][0], pt[pt0][0][1]+gap ]
        ptd[pt2][0] = [ pt[pt3][0][0], pt[pt3][0][1]+gap ]

        # cut edge of 1/20 % points
        offset = int(abs(pt[pt0][0][0] - pt[pt3][0][0])/20)

        if ptd[pt1][0][0] > ptd[pt2][0][0]:
            ptd[pt1][0][0] -= offset
            ptd[pt2][0][0] += offset
            ptd[pt0][0][0] -= offset
            ptd[pt3][0][0] += offset
        else:
            ptd[pt1][0][0] += offset
            ptd[pt2][0][0] -= offset
            ptd[pt0][0][0] += offset
            ptd[pt3][0][0] -= offset

    else: # x move < y move, move x
        if pt[pt0][0][0] < pt[pt1][0][0] and \
           pt[pt3][0][0] < pt[pt2][0][0]:
            gap = (-1)*gap
        ptd[pt1][0] = [ pt[pt0][0][0]+gap, pt[pt0][0][1] ]
        ptd[pt2][0] = [ pt[pt3][0][0]+gap, pt[pt3][0][1] ]

        # cut edge of 1/20 % points
        offset = int(abs(pt[pt0][0][1] - pt[pt3][0][1])/20)

        if ptd[pt1][0][1] > ptd[pt2][0][1]:
            ptd[pt1][0][1] -= offset
            ptd[pt2][0][1] += offset
            ptd[pt0][0][1] -= offset
            ptd[pt3][0][1] += offset

        else:
            ptd[pt1][0][1] += offset
            ptd[pt2][0][1] -= offset
            ptd[pt0][0][1] += offset
            ptd[pt3][0][1] -= offset


    return ptd

def get_4_outer_rect(pt):
    pt_rect=[]
    for i in range(4):
        pt_rect.append(get_outer_rect(pt,i))

    return pt_rect

def get_edge(shape,c,approx):
    pt_rect = get_4_outer_rect(approx)
    outer_pt0 = pt_rect[0]


    # Crop mask threshold image
    pts=[]
    for r in pt_rect:
        th0 = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        th_contour = copy.deepcopy(th0)

        th1 = cv2.fillPoly(th0, pts=[r], color=(255))
        cv2.drawContours(th_contour,c,-1,(255),thickness=1)
        th = cv2.bitwise_and(th0,th_contour)

        pt1 = cv2.findNonZero(th)[:,0]
        pt1 = cv2.transpose(pt1)
        pts.append(pt1)

        #im_draw('get_edge',th)

    return pts

# for scipy.optimize fit line
def f_1(x, K, B):
    return K*x + B

def get_4_points_by_fit_lines(pts,im=None):

    # 4 head points
    head_pts=[]
    k=[]
    b=[]

    for i in range(4):
        #j = (i+1) % 4

        # if k==infinit?
        if max(pts[i][0])-min(pts[i][0])<=3:
            K=None
            B = np.mean(pts[i][0])
        else:
            [K,B] = np.polyfit(pts[i][0],pts[i][1], 1)

        k.append(K)
        b.append(B)
        #[k1,b1] = np.polyfit(pts[i][0],pts[i][1], 1)
        #[k2,b2] = np.polyfit(pts[j][0],pts[j][1], 1)
        #k1, b1 = optimize.curve_fit(f_1, pts[i][0],pts[i][1])[0]
        #k2, b2 = optimize.curve_fit(f_1, pts[j][0],pts[j][1])[0]

        # check data
        #if im is not None:
        #    print ('k1={:0.2f},b1={:0.2f},k2={:0.2f},b2={:0.2f}'.format(k1,b1,k2,b2))
        #    im_draw_line(im,k1,b1)
        #    im_draw_pt(im,pts[i])

    for i in range(4):
        j = (i + 1) % 4
        k1 = k[i]
        b1 = b[i]
        k2 = k[j]
        b2 = b[j]
        if k1 is None:
            x = b1
            y = k2*b1 + b2
        elif k2 is None:
            x = b2
            y = k1*b2 + b1
        else:
            # k1 != k2, will be orthogonal
            x = (b2-b1)/(k1-k2)
            y = (b1*k2-b2*k1)/(k2-k1)

        head_pts.append([x,y])

    return head_pts


# from head_pts, find left-top index
# head_pts format is the return value of 'get_4_points_by_fit_lines'
def find_left_top(head_pts):
    z=[]
    for h in head_pts:
       z.append(h[0] + h[1])

    return z.index(min(z))


def find_rect_contour(img_name):
    bgtime = time.time()
    img = cv2.imread(img_name)
    #hi = int(img.shape[0]/2)
    #wi = int(img.shape[1]/2)
    #img = cv2.resize(img,(wi,hi))
    edtime = time.time()
    #print('read time={}'.format(edtime - bgtime))

    
    # enhance colour image, and get gray
    bgtime=edtime
    gray = enhance_img(img)
    edtime = time.time()
    #print('enhance img time={}'.format(edtime - bgtime))
    
    # get threshold image and max contour
    bgtime=edtime
    th,max_contour = get_max_threshold_img(gray,80,150)
    if th is None:
        return None
    edtime = time.time()
    #print('get threshold time={}'.format(edtime - bgtime))
    
    # get 4-corner points
    bgtime=edtime
    approx = approx_fit_quadrilateral(max_contour)
    if approx.shape != (4,1,2):
        return None # color error
    edtime = time.time()
    #print('approxite 4 corner time={}'.format(edtime - bgtime))

    # get outer quatrilateral
    bgtime=edtime
    points = get_edge(th.shape,max_contour,approx)
    edtime = time.time()
    #print('get 4 edges time={}'.format(edtime - bgtime))
    
    # get 4 points
    bgtime=edtime
    head_points = get_4_points_by_fit_lines(points,img)
    edtime = time.time()
    #print('get 4 points axis time={}'.format(edtime - bgtime))
    
    #th3 = np.dstack([th, th, th])
    if is_show or is_save:
        for i in range(4):
            j=(i+1)%4
            x0 = int(head_points[i][0])
            y0 = int(head_points[i][1])
            x1 = int(head_points[j][0])
            y1 = int(head_points[j][1])
            cv2.line(img,(x0,y0),(x1,y1),(0,255,0),1,4)

            round_x = round(head_points[i][0],4)
            round_y = round(head_points[i][1],4)
            coor = "{:0.4f} x {:0.4f}".format(round_x,round_y)
            cv2.putText(img,coor, (x0, y0-5), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 1)

        ### show HSV on corner
        msg = 'HSV: {:d}, {:d}, {:d} to {:d}, {:d}, {:d}'.format(
            hsv_low[0],hsv_low[1],hsv_low[2],
            hsv_high[0],hsv_high[1],hsv_high[2]
        )
        cv2.putText(img, msg, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    # show
    if is_show:
        im_draw('rect',img)

    # save marked
    if is_save:
        #if not os.path.isdir(save_dir):
        #    os.system('mkdir -p ' + save_dir)
        _,new_name = os.path.split(img_name)
        cv2.imwrite(save_dir + new_name,img)

    return head_points


if __name__ == '__main__':

    begintime = time.time()


    if len(sys.argv)==2:
        if sys.argv[1]=='-v' or sys.argv[1]=='--version':
            print('measure ver 1.0.3')
            sys.exit(0)
        elif sys.argv[1]=='-h' or sys.argv[1]=='--help':
            print('measure -h or measure --help to see this help')
            print('measure -v or measure --version to see version\n')
            print('measure <imagename|imagepath>  <show|data|save> [output_dir]')
            print('    imagename : one image name, suffix is jpg or JPG')
            print('    show : show image with label axis (x,y)')
            print('    data : output 4 points axis in console')
            print('    save : save analysis image to output dir you are given')
        sys.exit(0)
    elif len(sys.argv)>=3:
        imgname = sys.argv[1]
        is_show = sys.argv[2] =='show'
        is_save = sys.argv[2] =='save'
        if len(sys.argv)>=4:
            save_dir = sys.argv[3]
            if not os.path.isdir(save_dir):
                os.system('mkdir -p ' + save_dir)
                if os.path.isdir(save_dir):
                    print('can not create path: ' + save_dir)
                    exit(1)
            if save_dir[-1:] != '/' or save_dir[-1:] != '\\':
                save_dir = save_dir + '/'
        else:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
    else:
        print('usage: measure <imagename> [show|save|data]')
        print('more help: measure -h')
        sys.exit(1)


    ### read config hsv low and high
    
    configParser = configparser.RawConfigParser()
    configFilePath = r'm_config.txt'
    configParser.read(configFilePath, encoding="utf-8-sig")
    config_data = {}

    item = 'type'
    item_colours = configParser.get(item, 'colour')
    item_colours = item_colours.split((','))

    for item_colour in item_colours:
        item = item_colour
        tmp = configParser.get(item, 'low')
        tmp = tmp.split(',')
        hsv_low = np.array([int(int(tmp[0])/2), int(int(tmp[1])*2.55), int(int(tmp[2])*2.55)])

        tmp = configParser.get(item, 'high')
        tmp = tmp.split(',')
        hsv_high = np.array([int(int(tmp[0])/2+1), int(int(tmp[1])*2.55), int(int(tmp[2])*2.55)])

        colour_num=1
        if os.path.isfile(imgname):
            headpoints = find_rect_contour(imgname)
            if headpoints is not None:
                endtime = time.time()
                duration= endtime - begintime
                print(imgname + ' : ' + str(colour_num)+ ', calc time={:0.4f} sec'.format(duration))
                print(headpoints)
                sys.exit(0)

        
        elif os.path.isdir(imgname): # infinit loop for read image and analysis
            is_save = False
            is_data = True
            is_show = False
            imgpath = imgname
            
            ### batch image dir
            if imgpath[-1:]!='/' or imgpath[-1:]!='\\':
                img_dir = imgpath + '/'
            else:
                img_dir = imgpath

            while True:
                img_list = os.listdir(img_dir)
                for subimgname in img_list:
                    if subimgname[-4:]=='.jpg' or subimgname[-4:]=='.JPG':
                        headpoints = find_rect_contour(imgpath + subimgname)
                        if headpoints is not None:
                            endtime = time.time()
                            duration= endtime - begintime
                            print(subimgname + ' : ' + str(colour_num))
                            print('calc-time={:0.4f} sec'.format(duration))
                            print(headpoints)
                            print()
                            begintime = endtime
                            
                        if os.path.isfile(save_dir + subimgname):
                            shutil.remove(save_dir + subimgname)
                        shutil.move(imgpath + subimgname,save_dir + subimgname)
                    elif imgname=='end-program':
                        sys.exit(0)
                time.sleep(0.5)

    ### no found and exit
    print(imgname + ' : 0')
    print([[0,0],[0,0],[0,0],[0,0]])