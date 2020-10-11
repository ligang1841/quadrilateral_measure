### module Gap finder

import os, sys,glob
import cv2, imutils
import numpy as np
import configparser

### ================= global var ==========================
is_show = True
is_save = False
ori_im = None
#ori_im = cv2.imread('./gap/gap_sample.jpg')

### ================= private func ========================

def im_draw(im, title='no title',merge=False):
    if not is_show:
        return

    h = im.shape[0]
    w = im.shape[1]
    if w>800 or h >600:
        scalesize = max(w/800,h/600)
        w /= scalesize
        h /= scalesize
        w = int(w)
        h = int(h)
        im = cv2.resize(im,(w,h))
        if len(im.shape)==2:
            black = np.zeros((im.shape[0], im.shape[1]), np.uint8)
            im = np.dstack((black,im,black))
        im_bg = cv2.resize(ori_im,(w,h))
    else:
        im_bg = ori_im

    r=0.3
    if merge:
        img = cv2.addWeighted(im_bg,r,im,1.0-r,0)
        print('merge image show')
    else:
        img = im
    cv2.imshow(title, img)
    key = cv2.waitKey()
    cv2.destroyAllWindows()

    ### esc exit or other key go on
    if key == 27:
        sys.exit()
    elif key==ord('w'):
        cv2.imwrite(title[:-4]+'_sc.jpg', img)


### one channel to RGB, this channel to green
def im_to_green(green):
    black = np.zeros((green.shape[0], green.shape[1]), np.uint8)
    im = np.dstack((black, green, black))

    return im

### ================= process func ========================

### rgb img white_balance
def white_balance(im):
    r, g, b = cv2.split(im)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]

    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    return cv2.merge([b, g, r])


### enhance:
def enhance_img(im,thres=None):
    #im = white_balance(im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (5, 5), 0)

    ### contrast
    if thres is not None:
        im[im< thres[0]] = 0
        im[im>=thres[1]] = 255

    return im

### find contour
def canny_contour(img,thres):
    edge = cv2.Canny(img,thres[0],thres[1])
    #im_draw(edge,'after canny')

    return edge


### Sobel
#   dx = 0,1,2,... Derivative of x
#   dy = 0,1,2,... Derivative of y
#   ksize = 1,3,5,... odd number
def sobel_contour(img,ksize=3):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # convert to uint8
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    #im_draw(sobel,'sobel contour')

    return sobel


### get mono-img, and keep percent of dark pixel
def thresholded_img(img,percent=0.05):
    ### get histogram in hist shape=[h,w]
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_count = hist.sum()
    
    thres_hist = int(hist_count * percent)
    keep_count = 0
    
    for i in range(255):
        keep_count += hist[i].sum()
        if keep_count >=  thres_hist:
            break
    
    img[img>i]=0

    #im_draw(img,'thresholded')
    
    return img

def after_enhance(img,k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    im = cv2.dilate(img,kernel)
    #im = cv2.dilate(im,kernel)

    #im_draw(im,'after enhanced')
    return im


### get mono img, and find all contours in mask
def find_contours(img,maxcount=3):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # remake threshold image, with max contour and fill 255
    mask = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
    #contour_all = len(contours)
    count=0
    for con in contours:
        if count<maxcount:
            #cv2.fillPoly(mask, con, color=[0,255,0])
            #im_draw(mask, 'find contours')
            cv2.drawContours(mask,[con],-1,(0,255,0),-1)
        count+=1

    #im_draw(mask, 'find contours')
    return mask

def find_connection_domain(img):
    ### connection domain
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(img, connectivity=8)

    ###查看各个返回值
    # 连通域数量
    print('num_labels = ', num_labels)

    ###连通域的信息：对应各个轮廓的x、y、width、height和面积
    #print('st:  x      y     w     h     area')
    #print(stats)
    #print()

    # 连通域的中心点
    #print('centroids = ', centroids)

    # 每一个像素的标签1、2、3...，同一个连通域的标签是一致的
    #print('labels = ', labels)

    #im = img
    green = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    black = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(1, num_labels):
        if stats[i][4] > 100:
            mask = labels == i
            green[:,:][mask]=255
    green = find_contours(green)
    im = np.dstack((black, green, black))

    '''
    im = np.zeros((im.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        if stats[i][4]>100:
            mask = labels == i
            im[:, :, 0][mask] = 0 #np.random.randint(0, 255)
            im[:, :, 1][mask] = 255 #np.random.randint(0, 255)
            im[:, :, 2][mask] = 0 #np.random.randint(0, 255)
    '''

    #im_draw(im,'connection')

    return im


### batch dir images
def batch_images(img_path):
    ### read config
    conf = configparser.RawConfigParser()
    configFilePath = r'gap_config.txt'
    conf.read(configFilePath, encoding="utf-8-sig")
    #config_data = {}

    item='config'
    histpercent = float(conf.get(item, 'histpercent')) # 0.1 e.g.
    if img_path[-1:]!='/':
        img_path += '/'
    img_type = conf.get(item, 'img_type') # *.jpg or *.png
    ksize = int(conf.get(item, 'dilate_ksize')) # 1,3,5,7 ... default =3
    max_contours = int(conf.get(item, 'max_contours')) # 1,2,3 ... default =3

    global is_show, is_save
    is_show = bool(conf.get(item, 'is_review'))
    is_save = conf.get(item, 'is_save')


    global ori_im
    ilist = glob.glob(img_path + '*.' + img_type)
    for iname in ilist:
        ori_im = cv2.imread(iname)
        im_gray = enhance_img(ori_im)

        im_histed = thresholded_img(im_gray, histpercent)
        im_dilate = after_enhance(im_histed, ksize)
        im_contour= find_contours(im_dilate, max_contours)

        if is_show:
            im_draw(im_contour,iname)

        if is_save!='none':
            _, g, _ = cv2.split(im_contour)
            g[g>1] = 1

            if is_save=='mat':
                np.savetxt(iname[:-4] + '.mat',g,fmt='%2d')
            elif is_save=='rle':
                # save to RLE format
                col_num = g.shape[1]
                row_count=0
                txt=''
                for row in g:
                    row[0]=0
                    row[col_num-1]=0
                    count=0
                    for num in range(0,col_num-2):
                        if row[num]==0 and row[num+1]==1:
                            b=num
                        elif row[num]==1 and row[num+1]==0:
                            lens = num - b +1
                            txt += 'row={:d},begin={:d},len={:d}\n'.format(row_count, b, lens)
                    row_count +=1
                    print(row_count)
                if is_save=='rle':
                    with open(iname[:-4] +'.txt','w') as f:
                        f.write(txt)

            elif is_save=='npy': # binary
                np.save(iname[:-4] + '.npy',g)
            elif is_save=='npz': # compressed binary
                np.savez_compressed(iname[:-4]+'.npz',g)

### one image
def calc_image(img_info):
    iname = img_info[1]
    x = int(img_info[2])
    y = int(img_info[3])
    x_w = int(img_info[4])
    y_h = int(img_info[5])

    ### read config
    conf = configparser.RawConfigParser()
    configFilePath = r'gap_config.txt'
    conf.read(configFilePath, encoding="utf-8-sig")
    #config_data = {}

    item='config'
    histpercent = float(conf.get(item, 'histpercent')) # 0.1 e.g.
    img_type = conf.get(item, 'img_type') # *.jpg or *.png
    ksize = int(conf.get(item, 'dilate_ksize')) # 1,3,5,7 ... default =3
    max_contours = int(conf.get(item, 'max_contours')) # 1,2,3 ... default =3

    global is_show, is_save
    is_show = bool(conf.get(item, 'is_review'))
    is_save = conf.get(item, 'is_save')

    global ori_im
    ori_im = cv2.imread(iname)
    rect_im = ori_im[y:y_h,x:x_w]
    im_gray = enhance_img(rect_im)


    im_histed = thresholded_img(im_gray, histpercent)
    im_dilate = after_enhance(im_histed, ksize)
    im_contour = find_contours(im_dilate, max_contours)

    if is_show:
        im_draw(im_contour, iname)

    if is_save != 'none':
        _, g, _ = cv2.split(im_contour)
        g[g > 1] = 1

        if is_save == 'mat':
            np.savetxt(iname + '.mat', g, fmt='%2d')
        elif is_save == 'rle':
            # save to RLE format
            col_num = g.shape[1]
            row_count = 0
            txt = 'x1={:d},y1={:d},x2={:d},y2={:d}\n'.format(x,y,x_w,y_h)
            for row in g:
                row[0] = 0
                row[col_num - 1] = 0
                count = 0
                for num in range(0, col_num - 2):
                    if row[num] == 0 and row[num + 1] == 1:
                        b = num
                    elif row[num] == 1 and row[num + 1] == 0:
                        lens = num - b + 1
                        txt += 'row={:d},begin={:d},len={:d}\n'.format(row_count, b, lens)
                row_count += 1
                #print(row_count)
            if is_save == 'rle':
                txtname = '{:s}.{:d}x{:d}.{:d}x{:d}.txt'.format(iname[:-4], x, y, x_w, y_h)
                with open(txtname, 'w') as f:
                    f.write(txt)

        elif is_save == 'npy':  # binary
            np.save(iname + '.npy', g)
        elif is_save == 'npz':  # compressed binary
            np.savez_compressed(iname + '.npz', g)

        ### save added rect image
        add_rect_img = cv2.rectangle(ori_im,(x,y),(x_w,y_h),color=(0,0,255),thickness=2)
        rect_img_name = '{:s}.{:d}x{:d}.{:d}x{:d}.jpg'.format(iname[:-4], x, y, x_w, y_h)
        cv2.imwrite(rect_img_name, add_rect_img)


### ======= main =======
if __name__ == '__main__':
    if os.path.isfile(sys.argv[1]):
        calc_image(sys.argv) # have [img_name, x, y, x_w, y_h] input
    else:
        batch_images(sys.argv[1])