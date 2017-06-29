import cv2
import math
import numpy as np
import metrics

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark,kind):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/100),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    #print(indices.shape)
    indices = indices[imsz-numpx:,0]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = (atmsum / numpx)

    #pure water
    if kind==0:
        return A* 1.06 * 1.6 / 0.043
    #clean water
    if kind==1:
        return A* 1.06 * 2.46 / 0.151
    #castal
    if kind==2:
        return A* 1.06 * 5.24 / 0.398
    #
    if kind==3:
        return A* 1.06 * 8.41 / 2.19


def TransmissionEstimate(im,A,sz):
    #omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    #adjust
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            for k in range(res.shape[2]):
                if res[i,j,k]<0:
                    res[i,j,k]=0
                if res[i,j,k]>1:
                    res[i,j,k]=1
    return res

def recoverEnhancement(recover,kind):
    temp=recover.copy()
    if kind==0:
        temp= 1.4 * recover - 0.1
    if kind==1:
        temp = 1.5 * recover - 0.1
    if kind==2:
        temp = 2 * recover-0.1
    if kind==3:
        temp = 1.5 * recover

    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            for k in range(temp.shape[2]):
                if temp[i,j,k]<0:
                    temp[i,j,k]=0
                if temp[i,j,k]>1:
                    temp[i,j,k]=1
    return temp



def get_recover(img,size,kind):
    # trans to float
    I = img.astype('float64') / 255

    # get darkMap
    darkMap = DarkChannel(I, size)
    print(darkMap)

    # atmosphere light
    A = AtmLight(I, darkMap,kind)

    # transMap
    transMap_estimate = TransmissionEstimate(I, A, size)

    # transMap_refine
    transMap_refine = TransmissionRefine(img, transMap_estimate)

    # recover
    recover = Recover(I, transMap_refine, A, 0.1)


    #testing code
    cv2.imshow("original",img)
    #print ("darkMap:",darkMap)
    cv2.imshow(winname="darkmap", mat=darkMap)
    #print ("darkMap.shape:",darkMap.shape)
    
    #print ("A:",A)
    #print(A.shape)
    
    cv2.imshow("TransMap_estimate:", transMap_estimate)
    #print("shape of transMap_estimate:",transMap_estimate.shape)
    
    cv2.imshow("TransMap_refine:", transMap_refine)
    #print("shape of transMap_refine:",transMap_refine.shape)
    
    #print ("recover:",recover)
    cv2.imshow("recover", recover)

    #recover2 = (recover * 255).astype(np.uint8)
    #print(recover2)
    #cv2.imshow("recover2", recover2)

    recover3=recoverEnhancement(recover,kind)

    cv2.imshow("recover3",recover3)

    print(metrics.get_all_metrics(I))
    print(metrics.get_all_metrics(recover))
    print(metrics.get_all_metrics(recover3))


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #return recover