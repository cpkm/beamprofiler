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


from tkinter import *
from tkinter.filedialog import asksaveasfilename
from PIL import Image, ImageTk


class Application(Tk):
    """ GUI app to aid in the collection of beam profile data"""
    
    def __init__(self, master=None):
        """Initialize frame"""
        root = self.root = Tk.__init__(self,master)
        
        self.geometry('425x350')
        self.protocol("WM_DELETE_WINDOW", self.onClose)
        
        self.grid()
        self.createWidgets()
        self.lift()
        
    def createWidgets(self):
        """Create button, text, entry widgets"""
        
        self.camSelectLabel = Label(self, text = 'Select camera:')
        self.camSelectLabel.grid(row = 0, column = 0, columnspan = 1, sticky = W)
        
        self.cameraName = StringVar()
        self.cameraName.set('-Select-')
        camera_list = self.getCameras()
        
        self.camSelect = OptionMenu(self, self.cameraName, *camera_list)
        self.camSelect.grid(row = 0, column = 1, columnspan = 1, sticky = W)        
        
        self.imgAvLabel = Label(self, text = 'Image to average: ')
        self.imgAvLabel.grid(row=0, column=2, sticky = W)
        
        self.imgAvNum = StringVar()
        self.imgAvNum.set('100')        
        
        self.imgAvEntry = Entry(self, textvariable = self.imgAvNum)
        self.imgAvEntry.grid(row=0, column=3, columnspan=1, sticky = W)
        
        self.saveLabel = Label(self, text = 'Save location:')
        self.saveLabel.grid(row = 1, column = 0, columnspan = 1, sticky = W)
        
        self.saveDir = StringVar()
        self.saveDir.set('Enter save folder')
        
        self.saveDirBox = Entry(self, textvariable = self.saveDir)
        self.saveDirBox.grid(row=1, column=1, columnspan = 2, sticky=W)
        
        self.browseButton = Button(self, text = 'Browse', command = self.browseDir)
        self.browseButton.grid(row = 1, column = 3, columnspan = 1, sticky = W)
                
        self.capture = Button(self, text='Capture', command = self.onCapture)
        self.capture.grid(row=2,column=0, sticky = W)
        
        self.startPreviewButton = Button(self, text='Start Preview', command = self.startPreview)
        self.startPreviewButton.grid(row=2,column=1, sticky = W)

        self.stopPreviewButton = Button(self, text='Stop Preview', command = self.stopPreview)
        self.stopPreviewButton.grid(row=2,column=2, sticky = W)

        self.imgQueue = qu.Queue(maxsize=1000)        
        
        img = None
        self.previewPanel = Label(self, image = img)
        self.previewPanel.image = img
        self.previewPanel.grid(row=3, column=0, columnspan=4, sticky = W)
        
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
         
    def onClose(self):
        '''what to do when window is closed'''
        try:        
            self.stopPreview()   
        except:
            print('Could stop everything, destroying...')
                
        self.destroy()
        
    def onCapture(self):
        """Display message based on password input"""
        
        self.stopPreview()
        print('Stopped Preview')
                
        cap = cv2.VideoCapture(np.int(self.cameraName.get()))

        FRAMES_AVG = np.int(self.imgAvNum.get())

        i = 0
        while i<FRAMES_AVG:
            # Capture frame-by-frame
            ret, frame = cap.read()
    
            if frame is not None:
                if i == 0:
                    im = frame.astype(float)/255
        
                else:
                    im = (i*im + frame.astype(float)/255)/(i+1)

            i += 1


        cap.release()
        print('Image captured')

        im[:,:,[0,1,2]] = im[:,:,[2,1,0]]
        img = Image.fromarray(np.uint8(im*255))
        
        j = 0
        while os.path.exists(self.saveDir.get() + '%03d.jpeg' % j):
            j += 1
        filename = self.saveDir.get() + '%03d.jpeg' % j
        img.save(filename) #this depends on what I use to get the image
        print('Image saved')
        
        #self.startPreview(self.camera_name.get())
        #print('Yup started preview')
        
        
    def startPreview(self):
        '''start camera preview'''
        #print(self.stopEvent.is_set())
        self.stopEvent = None
        self.thread = None
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.stopEvent = threading.Event()

        self.cam = cv2.VideoCapture(np.int(self.cameraName.get()))
        
        self.thread.start()
        print('Preview started')
        
        #self.after(20, self.checkQueue)

    def stopPreview(self):
        
        try:
            self.stopEvent.set()
            print('Set thread stop signal')
            self.cam.release()
            print('Released camera')
            self.previewPanel.configure(image = None)
            print('Removed preview image')

        except:
            raise
        
       
    def videoLoop(self):
        
        try:
            stop_check = self.stopEvent.is_set()
            if not stop_check:
                # Capture frame-by-frame
                #print(stop_check)
                r,frame = self.cam.read()
                
                if r:
                    #print('got frame')
                    width = 400
                    frame_sized = cv2.resize(frame, (width, np.int(9*width/16)))  
                
                    # Display the resulting frame
                    #cv2.imshow(WINDOW_NAME,frame_sized)
                
                    frame_sized = cv2.cvtColor(frame_sized,cv2.COLOR_BGR2RGBA)
                    image = Image.fromarray(frame_sized)
                    img = ImageTk.PhotoImage(image)
                    
                    #self.imgQueue.put(img)
                    #self.after(20, self.videoLoop)
                    
                    self.previewPanel.configure(image = img)
                    self.previewPanel.image = img 
                
                self.previewPanel.after(10, self.videoLoop)
                
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
