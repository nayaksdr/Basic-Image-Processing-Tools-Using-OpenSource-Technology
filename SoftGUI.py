from tkinter import *
#import OpenFile 
from tkinter.filedialog import askopenfilename
#import Filter
import easygui
from tkinter import *
import requests
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter.filedialog import askopenfilename
import cv2
from matplotlib import pyplot as plt
import numpy as np
import PyPDF2 
from tkinter import messagebox
import tkinter as tk
import numpy as np
import matplotlib .pyplot as plt
import skimage.io as io
import webbrowser 

from copy import deepcopy

#=====================GUI==========================================#
class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master, relief=SUNKEN, bd=2)

        self.menubar = Menu(self)
#=============================File Menu =====================================#
        menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=menu)
        menu.add_command(label="Open",command=ImageSearch)
        menu.add_command(label="Exit",command= Quit)
#============================View Menu=======================================#
        menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=menu)
        menu.add_command(label="RGB Channel", command=RGB_Channel)
        menu.add_command(label="RGBtoHSV", command=RGBtoHSV)
        menu.add_command(label="CMY To RGB", command=cmykToRgb)
#==============================Tools==========================================#
        menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=menu)
        menu.add_command(label="Opening",command=Opening)
        menu.add_command(label="Closing",command=Closing)
        menu.add_command(label="Rotation",command=ImageRotation)
        menu.add_command(label="Dilation",command=Dilation)
        menu.add_command(label="Erosion",command=Erosion)
        menu.add_command(label="Edge Detaction",command=EdgeDet)
        menu.add_command(label="Histogram",command=Histplot)
        menu.add_command(label="HistEqualiz",command=HistEqualiz)
        
#=============================Filter============================================#
        menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Filter", menu=menu)
        menu.add_command(label="2D filter",command=Twodifil)
        menu.add_command(label="Gaussion",command=GaussianBlur)
        menu.add_command(label="Median",command=MedianFil)
        menu.add_command(label="Bilateral",command=BilateralFil)
        menu.add_command(label="Low pass",command=low_pass)
        menu.add_command(label="High pass",command=BilateralFil)
        menu.add_command(label="Laplacian",command=Laplacian)
#=============================Help=======================================#
        menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=menu)
        menu.add_command(label="About", command=About)
        menu.add_command(label="Document", command=Documetation)
        

        try:
            self.master.config(menu=self.menubar)
        except AttributeError:
            # master is a toplevel window (Python 1.4/Tkinter 1.63)
            self.master.tk.call(master, "config", "-menu", self.menubar)

        self.canvas = Canvas(self, bg="white", width=750, height=400,
                             bd=0, highlightthickness=0)
        self.canvas.pack()

   


#================================Image Openning=============================================#

def ImageSearch():
#Open a file chooser dialog and allow the user to select an input Image
      global image
      path = tk.filedialog.askopenfilename()
      if len(path)>0:
             image = cv2.imread(path)
             messagebox.showinfo("open image successfully",ImageSearch)

#=================================Quit Menu=================================================#
def Quit():
    
    print ('Clicked Exit')
    choice = messagebox.askquestion('Exit', 'Are you sure ?', icon="question")
    
    if choice == 'yes':
        root.destroy()
    else:
        print ('Session continued !')

#================================Image Enhancement Operation=================================#

#===================================Image Rotation=========================================#

def ImageRotation():
    try:
        
        img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rows,cols= img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        plt.subplot(121),plt.imshow(img),plt.title('Original Image')
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(dst),plt.title('Image Rotation')
        plt.xticks([]),plt.yticks([])
        plt.show()
       
    except:
        tkMessageBox.showinfo("Message","Please select the image")

#================================Image Threshold=================================#

def ImgThreshold():
    try:
        img = image
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
        ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        title = ['original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
        images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

        for i in range(6):
              plt.subplot(2,3,i+1),
              plt.imshow(images[i],'gray')
              plt.title(title[i])
              plt.xticks([]),plt.yticks([])
        plt.show()
        
    except:
           tkMassageBox.showinfo("Message","Please select the image")

#================================= 2D-Filter =======================================#
def Twodifil():
    try:
        img=image
        kernel =np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)
        plt.subplot(121),
        plt.imshow(img),
        plt.title("Original Image")
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),
        plt.imshow(dst),
        plt.title("2D-Filter")
        plt.xticks([]),plt.yticks([])
        plt.show()
    except:
             tkMassageBox.showinfo("Message","Please select the image")

#================================= Gaussian Filter =======================================#

def GaussianFil():
      try:
            img=image
            img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            
            blur=blur=cv2.GaussianBlur(img,(5,5),0)
            plt.subplot(121),plt.imshow(img),plt.title('Original')
            plt.xticks([]),plt.yticks([])
            plt.subplot(122),plt.imshow(blur),plt.title('GaussianFil')
            plt.xticks([]),plt.yticks([])
            plt.show()
                     
      except:               
            tkMassageBox.showinfo("Message","Please select the image")


#================================= Median Filter =======================================#
def MedianFil():
      try:
            img=image
            median = cv2.medianBlur(image ,5)
            plt.subplot(121),
            plt.imshow(img),
            plt.title("Original Image")
            plt.xticks([]),plt.yticks([])
            plt.subplot(122),
            plt.imshow(median),
            plt.title("median Filter")
            plt.xticks([]),plt.yticks([])
            plt.show()
            
      except:
              tkMassageBox.showinfo("Message","Please select the image")

#================================= Bilateral Filter =======================================#

def BilateralFil():
       try:
             img=image
             blur=cv2.bilateralFilter(img,9,75,75)
             plt.subplot(121),plt.imshow(img),plt.title('Original')
             plt.xticks([]),plt.yticks([])
             plt.subplot(122),plt.imshow(blur),plt.title(' blured')
             plt.xticks([]),plt.yticks([])
             plt.show()
       except:
            tkMassageBox.showinfo("Message","Please select the image")

#================================= Erosion =======================================#


def Erosion():
          try:
                
                  img=image
                  img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                  kernel = np.ones((5,5),np.uint8)
                  img_erosion= cv2.erode(img,kernel,iterations=1)
                  plt.subplot(121),
                  plt.imshow(img),
                  plt.title("Original Image")
                  plt.xticks([]),plt.yticks([])
                  plt.subplot(122),
                  plt.imshow(img_erosion),
                  plt.title("Erosion")
                  plt.xticks([]),plt.yticks([])
                  plt.show()
          except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= Low Pass =======================================#
def low_pass ():
             try:
                
                  img=image
    
                  data = np.array(im, dtype=float)
                  plot(data, 'Original')

# A very simple and very narrow highpass filter
                  kernel = np.array([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]])
                  highpass_3x3 = ndimage.convolve(data, kernel)
                  plot(highpass_3x3, 'Simple 3x3 Highpass')

# A slightly "wider", but sill very simple highpass filter 
                  kernel = np.array([[-1, -1, -1, -1, -1],
                                     [-1,  1,  2,  1, -1],
                                     [-1,  2,  4,  2, -1],
                                     [-1,  1,  2,  1, -1],
                                     [-1, -1, -1, -1, -1]])
                  highpass_5x5 = ndimage.convolve(data, kernel)
                  plot(highpass_5x5, 'Simple 5x5 Highpass')

# Another way of making a highpass filter is to simply subtract a lowpass
# filtered image from the original. Here, we'll use a simple gaussian filter
# to "blur" (i.e. a lowpass filter) the original.
                  lowpass = ndimage.gaussian_filter(data, 3)
                  gauss_highpass = data - lowpass
                  plot(gauss_highpass, r'Gaussian Highpass, $\sigma = 3 pixels$')
                  plt.show()
             except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= Low Pass =======================================#
def GaussianBlur ():
             try:
                
                  img=image
    
                  blur = cv2.GaussianBlur(img,(5,5),0)

                  plt.subplot(121),plt.imshow(img),plt.title('Original')
                  plt.xticks([]), plt.yticks([])
                  plt.subplot(122),plt.imshow(blur),plt.title('GaussianBlur')
                  plt.xticks([]), plt.yticks([])
                  plt.show()
             except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= Dilation =======================================#
                   
def Dilation():
    try:
        img=image
        img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        img_dilation= cv2.dilate(img,kernel,iterations=1)
        plt.subplot(121),
        plt.imshow(img),
        plt.title("Original Image")
        plt.xticks([]),plt.yticks([])
        plt.subplot(122),
        plt.imshow(img_dilation),
        plt.title("Dilation")
        plt.xticks([]),plt.yticks([])
        plt.show()
    except:
            tkMassageBox.showinfo("Message","Please select the image")

#================================= Opening =======================================#
def Opening():
          try:
                  img=image
                  img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                  kernel = np.ones((5,5),np.uint8)
                  opening= cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
                  plt.subplot(121),
                  plt.imshow(img),
                  plt.title("Original Image")
                  plt.xticks([]),plt.yticks([])
                  plt.subplot(122),
                  plt.imshow(opening),
                  plt.title("opening")
                  plt.xticks([]),plt.yticks([])
                  plt.show()
        
              
          except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= Laplacian =======================================#

def Laplacian():

             try:

                img=image
                laplacian = cv2.Laplacian(img,cv2.CV_64F)
                sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
                sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

                plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
                plt.title('Original'), plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
                plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
                plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
                plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

                plt.show()

             except:
                
                   tkMassageBox.showinfo("Message","Please select the image")
                   
#================================= Closing =======================================#

def Closing():
          try:
                  img=image
                  img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                  kernel = np.ones((5,5),np.uint8)
                  closing= cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
                  plt.subplot(121),
                  plt.imshow(img),
                  plt.title("Original Image")
                  plt.xticks([]),plt.yticks([])
                  plt.subplot(122),
                  plt.imshow(closing),
                  plt.title("closing")
                  plt.xticks([]),plt.yticks([])
                  plt.show()
          except:
                  tkMassageBox.showinfo("Message","Please select the image")

#================================= EdgeDet =======================================#
                  
def EdgeDet():
          try:
              img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
              edges = cv2.Canny(img,100,200)
              edges=cv2.Canny(img,100,200)
              plt.subplot(121),
              plt.imshow(img),
              plt.title("Original Image")
              plt.xticks([]),plt.yticks([])
              plt.subplot(122),
              plt.imshow(edges),
              plt.title("edge detection")
              plt.xticks([]),plt.yticks([])
              plt.show()
                                
          except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= Histplot =======================================#                   
def Histplot():
          try :
              img = image
              color = ("b","g","r")
              for i, col in enumerate(color):
                  histr = cv2.calcHist([img],[i],None,[256],[0,256])
                  plt.plot(histr,color=col)
                  plt.xlim([0,256])
              plt.show()

          except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= HistEqualiz =======================================#         

def HistEqualiz():
          try:

                img=image
                #img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                hist,bins=np.histogram(img.flatten(),256,[0,256])
                cdf=hist.cumsum()
                cdf_normalized = cdf*hist.max()/cdf.max()
                plt.plot(cdf_normalized,color='b')
                plt.hist(img.flatten(),256,[0,256],color='r')
                plt.xlim([0,256])
                plt.legend(('cdf','histogram'),loc='upper left')
                plt.show()
          except:
                   tkMassageBox.showinfo("Message","Please select the image")

#================================= Red,Green,Blue =======================================#    


def RGB_Channel():
        

              img=image
              red_channel = deepcopy(img)
              green_channel = deepcopy(img)
              blue_channel = deepcopy(img)
              red_channel [ : , : , 1] = 0
              red_channel [ : , : , 2] = 0

              green_channel [ : , : , 0] = 0
              green_channel [ : , : , 2] = 0

              blue_channel [ : , : , 0] = 0
              blue_channel [ : , : , 1] = 0

              fig, ax = plt.subplots (ncols=2, nrows =2 )

              ax [ 0 , 0 ].imshow(image)
              ax [ 0 , 0 ] .set_title('Original')

              ax [ 0 , 1 ].imshow(red_channel)
              ax [ 0 , 1 ] .set_title('Red_channel')

              ax [ 1 , 0 ].imshow(green_channel)
              ax [ 1 , 0 ] .set_title('Green_channel')

              ax [ 1 , 1 ].imshow(blue_channel)
              ax [ 1 , 1 ] .set_title('Blue_channel')

              plt.show()

def RGBtoHSV():

              img =image
              img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

              cv2.imshow('HSV Image',img_HSV)

              cv2.imshow('Hue Channel',img_HSV[ : , : , 0] )

              cv2.imshow('Saturation',img_HSV[ : , : , 1] )

              cv2.imshow('Value Channel',img_HSV[ : , : , 2] )

              cv2.waitkey(0)

def cmykToRgb() :
               img=image
               if image.mode == 'CMYK':
                   rgb_image = img.convert('RGB')

                   if image.mode == 'RGB':
                       cmyk_image = img.convert('CMYK')

              
def Documetation():
    helpDocumentation = r'C:/Users/sachi/OneDrive/Documents/doc.pdf'
    webbrowser.open_new(helpDocumentation)


def Documetation():
    helpDocumentation = r'C:/Users/sachi/OneDrive/Documents/doc.pdf'
    webbrowser.open_new(helpDocumentation)              
# importing required modules 


# creating a pdf file object 
##               pdfFile = "oc.pdf"
##
##               pdfRead = PyPDF2.PdfFileReader(pdfFile)
##               page = pdfRead.getPage(0)
##               pageContent = page.extractText()
##               print(pageContent)

#if __name__=="__main__":

#    Documetation()

def About():
    About1 = r'C:/Users/sachi/OneDrive/Documents/doc.pdf'
    webbrowser.open_new(About1)   

    



                   

root = Tk()
root.title("Design and Develop Image Processing software using Open source technology")

root.resizable(0,0)
app = GUI(root)
app.pack()

root.mainloop()
