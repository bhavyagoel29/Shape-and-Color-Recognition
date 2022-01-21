#!/usr/bin/env python
# coding: utf-8

# In[4]:
import argparse

import numpy as np
import cv2
from PIL import Image
import pandas as pd


# In[18]:


def co(in1,img,img_c,po,tak):
    con, hei = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in con:
        area= cv2.contourArea(i)
        if area > int(po):
            cv2.drawContours(img_c,i,-1,(255,0,255),7)
            per = cv2.arcLength(i,True)
            app = cv2.approxPolyDP(i,0.02*per,True)
            if len(app) == 3:
                r='Triangle'
            elif len(app) == 4:
                r = 'Square'
            elif len(app) == 5:
                r = 'Pentagon'
            elif len(app) == 6:
                r = 'Hexagon'
            elif len(app) == 8:
                r = 'Octagon'
            else:
                r = len(app)
            x_,y_,w,h = cv2.boundingRect(app)
            cv2.rectangle(img_c,(x_,y_),(x_+w,y_+h),(255,0,255))
            croped = tak.crop((x_,y_,x_+w,y_+h))
            wid, hiegh = croped.size
            p_val = croped.getpixel((wid/2,hiegh/2))
            cv2.putText(img_c,'Shape: '+str(r),(x_+1,y_-30),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,0),1)
            cv2.putText(img_c,'Color: '+str(getColorName(p_val[0],p_val[1],p_val[2])),(x_+1,y_-15),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,0),1)


# In[15]:


def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname


# In[16]:


index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# In[17]:


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--mode', required=True, help="mode of input")
args = vars(ap.parse_args())
mode = args['mode']

if mode == 'camera':
    vid = cv2.VideoCapture(0)  
    while(True):
        ret, img = vid.read()
        img_c = img.copy()
        t1 = 157
        t2 = 118
        po = 1000
        kernel = np.ones([5,5])
        im = cv2.GaussianBlur(img,(7,7),1)
        img0 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im1 = cv2.Canny(img0,t1,t2)
        h = cv2.dilate(im1,kernel,7)
        cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        co(img,h,img_c,po,pil_im)
        cv2.imshow('VISION', img_c)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
elif mode == 'image':
    loca = str(input("Enter location of file here: "))
    img = cv2.imread(loca)  
    img_c = img.copy()
    t1 = 157
    t2 = 118
    po = 1000
    kernel = np.ones([5,5])
    im = cv2.GaussianBlur(img,(7,7),1)
    img0 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im1 = cv2.Canny(img0,t1,t2)
    h = cv2.dilate(im1,kernel,7)
    cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    co(img,h,img_c,po,pil_im)
    cv2.imshow('VISION', img_c)
    while(1):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()           
            
elif mode == 'video':
    loca = str(input("Enter location of file here: "))
    vid = cv2.VideoCapture(loca)  
    while(True):
        ret, img = vid.read()
        img_c = img.copy()
        t1 = 157
        t2 = 118
        po = 1000
        kernel = np.ones([5,5])
        im = cv2.GaussianBlur(img,(7,7),1)
        img0 = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im1 = cv2.Canny(img0,t1,t2)
        h = cv2.dilate(im1,kernel,7)
        cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        co(img,h,img_c,po,pil_im)
        cv2.imshow('VISION', img_c)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

# In[ ]:




