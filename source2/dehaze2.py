import cv2;
import math;
import numpy as np;

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

    return A*1.06*8.41/2.19



def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
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

    A = ((atmsum / numpx)*1.06*8.41)/2.19
    return A


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


if __name__ == '__main__':
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = './image/H170.jpg'


    def nothing(*argv):
        pass


    src = cv2.imread(fn);
    #compute the fraquent t
    A1=getMostFrequent(src)/255
    print("A1:",A1)


    print("src:",src)
    I = src.astype('float64') / 255;
    te1=TransmissionEstimate(I,A1,15)
    t1=TransmissionRefine(src,te1)

    cv2.imshow("te1", te1)
    cv2.imshow("t1", t1);

    dark = DarkChannel(I, 15);
    print(dark.dtype)
    print(dark.shape)
    A2 = AtmLight(I, dark);
    print("A2:",A2)
    te2 = TransmissionEstimate(I, A2, 15);
    t2 = TransmissionRefine(src, te2);
    t=(t1+t2)/2
    J = Recover(I, t, A2, 0.1);

    cv2.imshow("dark", dark);
    cv2.imshow("te2", te2)
    cv2.imshow("t2", t2);
    cv2.imshow('I', src);
    cv2.imshow('J', J);
    # cv2.imwrite("./image/J.png",J*255);
    cv2.waitKey();