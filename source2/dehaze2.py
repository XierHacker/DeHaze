import cv2
import math
import numpy as np


#---------------img statistics-----------------#
#get most frequently light
def getMostFrequent(img):
    A=np.zeros(shape=[1,3])
    g_array=np.zeros(shape=(256,))
    b_array=np.zeros(shape=(256,))
    r_array=np.zeros(shape=(256,))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            g_array[img[row,col,0]]+=1
            b_array[img[row, col, 1]] += 1
            r_array[img[row, col, 2]] += 1

    g_intensity=np.argmax(g_array)
    b_intensity=np.argmax(b_array)
    r_intensity=np.argmax(r_array)

    A[0,0]=g_intensity
    A[0,1]=b_intensity
    A[0,2]=r_intensity

    return A/255

def getMostBright(img):
    A = np.zeros(shape=(1, 3))
    height = img.shape[0]
    width = img.shape[1]
    total_pixel = height * width
    # print(total_pixel)
    numpx = int(max(math.floor(total_pixel / 1000), 1))
    # print(numpx)

    g_img = img[:, :, 0]
    b_img = img[:, :, 1]
    r_img = img[:, :, 2]

    g_img_vector = g_img.flatten()
    g_img_vector.sort()
    g_img_max = g_img_vector[-numpx:]
    g_average = g_img_max.mean()

    b_img_vector = b_img.flatten()
    b_img_vector.sort()
    b_img_max = b_img_vector[-numpx:]
    b_average = b_img_max.mean()

    r_img_vector = r_img.flatten()
    r_img_vector.sort()
    r_img_max = r_img_vector[-numpx:]
    r_average = r_img_max.mean()

    A[0, 0] = g_average
    A[0, 1] = b_average
    A[0, 2] = r_average
    return A

def getMostBrightByDarkMap(im,dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    print(indices.shape)
    indices = indices[imsz - numpx:, 0]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = (atmsum / numpx)
    return A

#-----------------------------------------------------#


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark,kind=0):
    A1=getMostBrightByDarkMap(im,dark)
    A2=getMostBright(im)

    A1=A1*1.06*8.41/2.19
    A2 = A2 * 1.06 * 8.41 / 2.19
    return A1,A2


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);

    return t;


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res
