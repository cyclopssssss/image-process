import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import filters
import tkinter as tk
lula_gray = cv2.imread("./figcode/lula.jpg",0)
lula = cv2.cvtColor(cv2.imread("./figcode/lula.jpg",1),cv2.COLOR_BGR2RGB)
lula_noise = cv2.cvtColor(cv2.imread("./figcode/lula.jpg"),cv2.COLOR_BGR2RGB)


def show_original_lula():
    plt.figure("lula_gray")
    plt.imshow(lula_gray, cmap ='gray')
    plt.show()
    return
#为直方图以及均衡化操作


img = np.array(lula_gray)
img1 = np.array(lula_noise)
arr = img.flatten()
equ = cv2.equalizeHist(img)
equr = equ.flatten()
res1 = cv2.resize(lula, None, fx=2, fy=2)


def show_hist():
    plt.figure("hist")
    arr = img.flatten()
    plt.hist(arr, bins=256, density=True, stacked=True, facecolor='green', alpha=0.75)
    plt.show()
    return


def show_equlula():
    equ = cv2.equalizeHist(img)
    plt.figure("equ")
    plt.imshow(equ, cmap ='gray')
    plt.show()
    return


def show_equhist():
    plt.figure("eqhist")
    equr = equ.flatten()
    plt.hist(equr, bins=256, density=True, stacked=True, facecolor='green', alpha=0.75)
    plt.show()
    return

#图像几何变换


def fig_resize():
    res1 = cv2.resize(lula, None, fx=2, fy=2)
    plt.figure("放大2x")
    plt.imshow(res1)
    plt.show()
    return


def fig_transform():
    flip_horizental = cv2.flip(lula, 1)
    flip_vertical = cv2.flip(lula, 0)
    flip_both = cv2.flip(lula, -1)
    plt.figure("flip_horizental")
    plt.imshow(flip_horizental)
    plt.figure("flip_vertical")
    plt.imshow(flip_vertical)
    plt.figure("flip_both")
    plt.imshow(flip_both)
    plt.show()
    return

#图像加噪及滤波


def fig_guassi_noise():
    gaussi = skimage.util.random_noise(img1, mode='gaussian', seed=None, clip=True)
    salt_pepper = skimage.util.random_noise(img1, mode='s&p', seed=None, clip=True)
    plt.figure("guassi")
    plt.imshow(gaussi)
    plt.show()
    return


def fig_salt_pepper_noise():
    gaussi = skimage.util.random_noise(img1, mode='gaussian', seed=None, clip=True)
    salt_pepper = skimage.util.random_noise(img1, mode='s&p', seed=None, clip=True)
    plt.figure("salt_pepper")
    plt.imshow(salt_pepper)
    plt.show()
    return

def guass_filter():
    gaussi = skimage.util.random_noise(img1, mode='gaussian', seed=None, clip=True)
    salt = skimage.util.random_noise(img1, mode='salt', seed=None, clip=True)
    guassfilter = filters.gaussian(salt,sigma=1, multichannel=0.6)
    plt.figure("guassfilter")
    plt.imshow(guassfilter)
    plt.show()
    return


window = tk.Tk()
window.title("图像处理")
window.geometry("300x300")

a = tk.Button(window, text='原始图像', font=('Arial', 12), width=10, height=1, command=show_original_lula).place(x=0, y=0, anchor='nw')
b = tk.Button(window, text='直方图', font=('Arial', 12), width=10, height=1, command=show_hist).place(x=100, y=0, anchor='nw')
c = tk.Button(window, text='均衡化后图像', font=('Arial', 12), width=10, height=1, command=show_equlula).place(x=200, y=0, anchor='nw')
d = tk.Button(window, text='均衡直方图', font=('Arial', 12), width=10, height=1, command=show_equhist).place(x=0, y=100, anchor='nw')
e = tk.Button(window, text='图像缩放', font=('Arial', 12), width=10, height=1, command=fig_resize).place(x=100, y=100, anchor='nw')
f = tk.Button(window, text='几何变换', font=('Arial', 12), width=10, height=1, command=fig_transform).place(x=200, y=100, anchor='nw')
c1 = tk.Checkbutton(window, text='高斯噪声', onvalue=1, offvalue=0,command=fig_guassi_noise).place(x=0, y=230, anchor='nw')
c2 = tk.Checkbutton(window, text='椒盐噪声', onvalue=1, offvalue=0,command=fig_salt_pepper_noise).place(x=0, y=260, anchor='nw')
g = tk.Button(window, text='图像加噪', font=('Arial', 12), width=10, height=1).place(x=0, y=200, anchor='nw')
h = tk.Button(window, text='图像去噪', font=('Arial', 12), width=10, height=1, command=guass_filter).place(x=100, y=200, anchor='nw')
window.mainloop()
