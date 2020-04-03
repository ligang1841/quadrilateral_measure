import os,sys
import cv2
import numpy as np
import imutils
import copy

#from sklearn import svm
#from sklearn import svm, datasets


img_dir = './image/'
save_dir = '/dev/shm/measure/'

is_debug = True
is_save = False

def im_draw(title,im):
    if not is_debug:
        return

    h = im.shape[0]
    w = im.shape[1]
    if w>1920 or h >1080:
        scalesize = min(w/1920,h/1080)
        w /= scalesize
        h /= scalesize
        w = int(w)
        h = int(h)
    im = cv2.resize(im,(w,h))
    cv2.imshow(title, im)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    if key == 27:
        sys.exit()

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

def enhance_colour(im):


    # rgb:
    #lower_rgb = np.array([60, 150, 180])
    #upper_rgb = np.array([150, 210, 220])
    #mask = cv2.inRange(im,lower_rgb,upper_rgb)

    # hsv: 180,255,255 (gimp:360,100,100)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([80,50,120])
    upper_hsv = np.array([110,200,240])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    #masked_img = cv2.bitwise_and(im, im, mask = mask)

    return mask


def enhance_img(im):
    #im_draw('origin',im)
    img = cv2.GaussianBlur(im, (5, 5), 0)
    im_gray = enhance_colour(im)
    im_gray = im_gray.astype(np.uint8)

    #im_draw('enhance_img',im_gray)
    return im_gray


def get_max_contour(im_th):
    # get max contour
    contours = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_c = max(contours, key=cv2.contourArea)
    return max_c


def approx_fit_quadrilateral(contour):
    # fit quadrilateral
    epsilon = 0.1*cv2.arcLength(contour,True)
    approx_points = cv2.approxPolyDP(contour,epsilon,True)
    return approx_points


def get_max_threshold_img(im_gray,l1,l2): # 100,150
    _,thres = cv2.threshold(im_gray,l1,255,cv2.THRESH_BINARY)
    thres = np.where(thres<l2, 0, thres)

    #im_draw('thres',thres)

    # get max contour
    c = get_max_contour(thres)

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
    img = cv2.imread(img_dir + img_name)
    #print(img_name)

    # enhance colour image, and get gray
    gray = enhance_img(img)

    # get threshold image and max contour
    th,max_contour = get_max_threshold_img(gray,10,250)

    # get 4-corner points
    approx = approx_fit_quadrilateral(max_contour)
    #im_draw_rect(img,approx)

    # get outer quatrilateral
    points = get_edge(th.shape,max_contour,approx)

    # get 4 points
    head_points = get_4_points_by_fit_lines(points,img)

    # get left-top-index
    #i = find_left_top(head_points)
    #print(i)

    #th3 = np.dstack([th, th, th])
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
        cv2.putText(img,coor, (x0, y0-5),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # show
    im_draw('rect',img)

    # save marked
    if is_save:
        cv2.imwrite(save_dir + img_name,img)

    return head_points



if __name__ == '__main__':
    print('params: <image-folder> [save-folder]')
    print('  example: measure ./image ./output')
    print('  press \'space\' to view next image')
    print('  press \'esc\' to exit')

    if len(sys.argv) < 2:
        print("please input image (.jpg) dir")
        sys.exit(1)

    img_dir = sys.argv[1]

    if os.path.isdir(img_dir):
        if img_dir[-1:] != '/':
            img_dir += '/'
    else:
        print(img_dir + ' not a directory')
        sys.exit(2)

    if len(sys.argv) >= 3:
        save_dir = sys.argv[2]
        if save_dir[-1:] != '/':
            save_dir += '/'
        if not os.path.isdir(save_dir):
            print(save_dir + ' is not a folder')
            sys.exit(3)
        is_save = True


    img_list = os.listdir(img_dir)
    if not os.path.exists(save_dir):
        os.system('mkdir ' + save_dir)
    else:
        os.system('rm ' + save_dir + '* 2>/dev/null')


    for fname in img_list:
        if fname[-4:]=='.jpg' or fname[-4:]=='.JPG':
            find_rect_contour(fname)
