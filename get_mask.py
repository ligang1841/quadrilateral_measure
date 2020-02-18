import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import imutils
import math
import copy

from sklearn import svm
from sklearn import svm, datasets


img_dir = './image/'
save_dir = '/dev/shm/measure/'

def im_draw(title,im):
    im = cv2.resize(im,(im.shape[1],im.shape[0]))
    cv2.imshow(title, im)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    if key == 27:
        exit()

def enhance_img(im):
    # median filter
    im = cv2.medianBlur(im,9)
    im = im.astype(np.uint16)

    # enhance read
    red = im[:,:,2]
    green = im[:,:,1]
    blue = im[:,:,0]

    im_gray = 3*red - green - blue - (green-blue)
    im_gray = np.where(im_gray <0, 0, im_gray)
    im_gray = im_gray//3
    im_gray = im_gray.astype(np.uint8)

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

    # get max contour
    c = get_max_contour(thres)

    # apprex fit quadrilateral
    approx_pt = approx_fit_quadrilateral(c)

    # remake threshold image, with max contour and fill 255
    th_new = np.zeros((thres.shape[0],thres.shape[1]),dtype=np.uint8)
    th_new = cv2.fillPoly(th_new, pts=[approx_pt], color=(255))
    thres = cv2.bitwise_or(th_new,thres)

    return thres,c

def get_outer_rect(pt,pt0):
    ptd = copy.deepcopy(pt)

    pt1 = (pt0+1) % 4
    pt2 = (pt1+1) % 4
    pt3 = (pt2+1) % 4

    gap = 30


    # pt0 to pt3 line check
    ck = [ pt[pt0][0][0] - pt[pt3][0][0],
           pt[pt0][0][1] - pt[pt3][0][1] ]


    # make new pt1 and pt2
    if abs(ck[0]) > abs(ck[1]): # x move > y move, move y
        if pt[pt0][0][1] < pt[pt1][0][1] and \
           pt[pt3][0][1] < pt[pt2][0][1]:
            gap = (-1)*gap
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
            ptd[pt3][0][1] -= offset

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

        #im_draw('con',th)

    return pts


def get_4_points_by_fit_lines(pts):

    # 4 head points
    head_pts=[]

    for i in range(4):
        j = (i+1) % 4
        [k1,b1] = np.polyfit(pts[i][0],pts[i][1], 1)
        [k2,b2] = np.polyfit(pts[j][0],pts[j][1], 1)

        #if k2!=k1: # means 2 lines parallel,will not happened
        # solve x,y
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

    # enhance colour image, and get gray
    gray = enhance_img(img)

    # get threshold image and max contour
    th,max_contour = get_max_threshold_img(gray,100,150)
    # get 4-corner points
    approx = approx_fit_quadrilateral(max_contour)

    # get outer quatrilateral
    points = get_edge(th.shape,max_contour,approx)

    # get 4 points
    head_points = get_4_points_by_fit_lines(points)

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
        cv2.line(img,(x0,y0),(x1,y1),(0,255,0),1,8)

        round_x = round(head_points[i][0],4)
        round_y = round(head_points[i][1],4)
        coor = "{:0.4f} x {:0.4f}".format(round_x,round_y)
        cv2.putText(img,coor, (x0, y0-5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    #im_draw('rect',img)

    cv2.imwrite(save_dir + img_name,img)

    return head_points



if __name__ == '__main__':
    img_list = os.listdir(img_dir)

    if not os.path.exists(save_dir):
        os.system('mkdir ' + save_dir)
    else:
        os.system('rm ' + save_dir + '*')

    for fname in img_list:
        find_rect_contour(fname)
