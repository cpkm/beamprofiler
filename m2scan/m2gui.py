# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:50:02 2014

@author: cpkmanchee
"""

import os
import numpy as np
import queue as qu
import cv2
import matplotlib.pyplot as plt
import threading
import time
import tkinter as tk

from tkinter.filedialog import asksaveasfilename
from PIL import Image, ImageTk


class Application(tk.Tk):
    """ GUI app to aid in the collection of beam profile data"""
    
    def __init__(self, master=None):
        """Initialize frame"""
        self.root = tk.Tk.__init__(self,master)
        
        self.geometry('475x375')
        self.protocol("WM_DELETE_WINDOW", self.onClose)
        
        self.grid()
        self.createWidgets()
        self.lift()
        
    def createWidgets(self):
        """Create button, text, entry widgets"""
        
        self.camSelectLabel = tk.Label(self, text = 'Select camera:')
        self.camSelectLabel.grid(row = 0, column = 0, columnspan = 1, sticky = 'W')
        
        self.cameraName = tk.StringVar()
        self.cameraName.set('-Select-')
        camera_list = self.getCameras()
        
        self.camSelect = tk.OptionMenu(self, self.cameraName, *camera_list)
        self.camSelect.config(width=8)
        self.camSelect.grid(row = 0, column = 1, columnspan = 1, sticky = 'EW')        
        
        self.imgAvLabel = tk.Label(self, text = 'Image to average: ')
        self.imgAvLabel.grid(row=0, column=2, sticky = 'W')
        
        self.imgAvNum = tk.StringVar()
        self.imgAvNum.set('10')        
        
        self.imgAvEntry = tk.Entry(self, textvariable = self.imgAvNum, width=10)
        self.imgAvEntry.grid(row=0, column=3, columnspan=1, sticky = 'W')
        
        self.saveLabel = tk.Label(self, text = 'Save location:')
        self.saveLabel.grid(row = 1, column = 0, columnspan = 1, sticky = 'W')
        
        self.saveDir = tk.StringVar()
        #home = os.path.expanduser("~")
        #self.saveDir.set(os.path.join(home, 'default_image'))
        self.saveDir.set(os.path.join('default_image','default_image'))
        
        self.saveDirBox = tk.Entry(self, textvariable = self.saveDir)
        self.saveDirBox.grid(row=1, column=1, columnspan = 2, sticky = 'W')
        
        self.browseButton = tk.Button(self, text = 'Browse', command = self.browseDir)
        self.browseButton.grid(row = 1, column = 3, columnspan = 1, sticky = 'W')
                
        self.capture = tk.Button(self, text='Capture', command = self.onCapture)
        self.capture.grid(row=2,column=0, sticky = 'W')
        
        self.startPreviewButton = tk.Button(self, text='Start Preview', command = self.startPreview)
        self.startPreviewButton.grid(row=2,column=1, sticky = 'W')

        self.stopPreviewButton = tk.Button(self, text='Stop Preview', command = self.stopPreview)
        self.stopPreviewButton.grid(row=2,column=2, sticky = 'W')

        #the next bit is for sensor saturation detection... not implemented yet
        self.resolutionText = tk.Label(self, text = 'Sensor resolution:')
        self.resolutionText.grid(row=3, column=0, columnspan=1, sticky='W')

        self.sensorRes = tk.DoubleVar()
        self.sensorRes.set(8)
        sensorList = np.array([8,10,12,14,16])
        self.resolutionMenu = tk.OptionMenu(self, self.sensorRes,*sensorList)
        self.resolutionMenu.grid(row=3, column=1, sticky='W')
        
        self.rSat = tk.Label(self, text='R', relief='raised', width = 5)
        self.rSat.grid(row=3, column=2, columnspan=2, sticky='W')
        self.gSat = tk.Label(self, text='G', relief='raised', width = 5)
        self.gSat.grid(row=3, column=2, columnspan=2)
        self.bSat = tk.Label(self, text='B', relief='raised', width = 5)
        self.bSat.grid(row=3, column=2, columnspan=2, sticky='E')
        
        self.bgColor = self.rSat.cget('bg')
        
        #end of sensor saturation detection

        self.imgQueue = qu.Queue(maxsize=100)        
        
        img = None
        self.previewPanel = tk.Label(self, image = img)
        self.previewPanel.image = img
        self.previewPanel.grid(row=4, column=0, columnspan=4, sticky = 'W')
        
        self.statusText = tk.StringVar()
        self.statusText.set('')
        self.statusBar = tk.Label(self, textvariable = self.statusText)
        self.statusBar.grid(row=5, column=0, columnspan=4, sticky = 'W')
    
    '''    
    def checkQueue(self):
        
      try:
         data = self.imgQueue.get_nowait()
         self.previewPanel.configure(image = data)
         self.previewPanel.image = data
      except qu.Empty:
         pass
      
      # make another check
      if not self.stopEvent.is_set():
          self.after_idle(self.checkQueue)
    '''

    def onClose(self):
        '''what to do when window is closed'''
        try:        
            self.stopPreview()   
        except:
            pass
                    
        self.destroy()
        
    
    def onCapture(self):
        """Display message based on password input"""
        
        t0 = time.time()

        self.stopPreview()

        w = 1280
        h = 720        
        
        cap = cv2.VideoCapture(np.int(self.cameraName.get()))
        #set image width, height
        cap.set(3,w)
        cap.set(4,h)


        
        FRAMES_AVG = np.int(self.imgAvNum.get())

        i = 0
        im=np.zeros((h,w,3))
        
        t1 = time.time()
        print(t1-t0)
        
        while i<FRAMES_AVG:
            # Capture frame-by-frame

            ret, frame = cap.read()

            if ret:
    
                im = (i*im + frame.astype(float)/255)/(i+1)
            '''
            if frame is not None:
                if i == 0:
                    #first frame
                    im = frame.astype(float)/255
        
                else:
                    #average subsequent frames
                    im = (i*im + frame.astype(float)/255)/(i+1)
            '''

            i += 1


        cap.release()
        self.statusText.set('Image captured')
        #print('Image captured')
        t2 = time.time()
        print(t2-t0, t2-t1)
        
        self.startPreview()

        im[:,:,[0,1,2]] = im[:,:,[2,1,0]]
        img = Image.fromarray(np.uint8(im*255))
        
        j = 0
        if not os.path.exists(os.path.dirname(self.saveDir.get())):
            os.makedirs(os.path.dirname(self.saveDir.get()))
        while os.path.exists(self.saveDir.get() + '%03d.jpeg' % j):
            j += 1
        filename = self.saveDir.get() + '%03d.jpeg' % j
        img.save(filename) #this depends on what I use to get the image
        #print('Image saved')
        
        #self.stopPreview() 
        #self.startPreview()

        t3 = time.time()
        print(t3-t0, t3-t2)
        self.statusText.set('Image saved')
        
        
    def startPreview(self):
        '''start camera preview'''
        #print(self.stopEvent.is_set())
        self.stopEvent = None
        self.thread = None
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.stopEvent = threading.Event()

        w = 1280
        h = 720

        self.cam = cv2.VideoCapture(np.int(self.cameraName.get()))
        self.cam.set(3, w)
        self.cam.set(4, h)
        
        self.thread.start()
        self.statusText.set('Preview started')
        
        #self.after(20, self.checkQueue)

    def stopPreview(self):
        
        try:
            self.stopEvent.set()
            #self.statusText.set('Set thread stop signal')
            self.cam.release()
            #self.statusText.set('Released camera')
            self.previewPanel.configure(image = None)
            self.previewPanel.image = None
            #self.statusText.set('Removed preview image')
            self.statusText.set('Stopped Preview')

        except:
            raise
        
       
    def videoLoop(self):
        
        try:
            #stop_check = self.stopEvent.is_set()
            while not self.stopEvent.wait(0.1):
                # Capture frame-by-frame

                r,frame = self.cam.read()
                
                if r:
                    #print('got frame')
                    width = 400
                    frame_sized = cv2.resize(frame, (width, np.int(9*width/16)))  
                
                    # Display the resulting frame
                    #cv2.imshow(WINDOW_NAME,frame_sized)
                
                    frame_sized = cv2.cvtColor(frame_sized,cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_sized)
                    img = ImageTk.PhotoImage(image)
                    
                    #self.imgQueue.put(img)
                    #self.after(20, self.videoLoop)
                    
                    self.previewPanel.configure(image = img)
                    self.previewPanel.image = img
                    #print(frame.shape)
                    sat_det = self.checkChannelSat(frame_sized)
                    self.setChannelSat(sat_det)
                
                #self.previewPanel.after(10, self.videoLoop)
                
        except AssertionError:
            #print('Some dumb AssertionError')
            raise
            
        except RuntimeError:
            #print('WTF, why a runtime error?')
            raise
        
        except:
            #print('Probably some stupid cv2 error... threading... yada, yada, yada')
            raise
#        cv2.waitKey(1)

        
    def getCameras(self):
        '''get list of available cameras'''

        camNum = 0
        camera_list =[]
        while camNum < 3:
            cam = cv2.VideoCapture(camNum)
            open_bool = cam.isOpened()
        
            if open_bool:
                camera_list.append(camNum)
        
            camNum += 1   
            
        return camera_list
        
    def browseDir(self):
        '''select save directory'''
        directory = asksaveasfilename()
        self.saveDir.set(directory)
        
    
    def checkChannelSat(self,im):
        '''
        check preview image channel satursation
        '''
        bits = 8
        satlim = 0.001
        sat_det = np.zeros(im.shape[2])
        
        for ind,_ in enumerate(sat_det):
        
            sat_ratio = (im[:,:,ind] >= 2**bits-1).sum()/(im[:,:,ind] != 0).sum()

            if sat_ratio <= satlim:
                sat_det[ind] = 0
            else:
                sat_det[ind] = 1
                
        return sat_det
    
    
    def setChannelSat(self, sat_det):
        '''
        set appropriate rgb indicator label for saturation
        sat_det is 1x3 boolean, 0=not sat, 1=sat
        '''
        
        channels = [self.rSat, self.gSat, self.bSat]
        cols = ['red', 'green', 'blue']

        for ind, ch in enumerate(channels):
            if sat_det[ind]:
                ch.configure(bg=cols[ind])
            else:
                ch.configure(bg=self.bgColor)
                
            
        
        
'''
root = Tk()
root.title("Beam Profile Collection")
root.geometry('425x350')

app = Application(root)

root.mainloop()
'''

app = Application()
app.title("Beam Profile Collection")
app.mainloop()
