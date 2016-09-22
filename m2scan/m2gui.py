# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:50:02 2014

@author: cpkmanchee
"""

import os
import numpy as np
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
        
    def createWidgets(self):
        """Create button, text, entry widgets"""
        
        self.thread = None
        self.stopEvent = None  

        self.camSelectLabel = Label(self, text = 'Select camera:')
        self.camSelectLabel.grid(row = 0, column = 0, columnspan = 1, sticky = W)
        
        self.camera_name = StringVar()
        self.camera_name.set('-Select-')
        camera_list = self.getCameras()
        
        self.camSelect = OptionMenu(self, self.camera_name, *camera_list, command = self.startPreview)
        self.camSelect.grid(row = 0, column = 1, columnspan = 1, sticky = W)        
        
        self.cameraLabel = Label(self, textvariable = self.camera_name)
        self.cameraLabel.grid(row=0, column=2, columnspan = 1, sticky = W)
        
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
        
        img = None
        self.previewPanel = Label(self, image = img)
        self.previewPanel.image = img
        self.previewPanel.grid(row=3, column=0, columnspan=4, sticky = W)
        
         
         
    def onClose(self):
        '''what to do when window is closed'''
        self.stopPreview()
        self.destroy()
        
    def onCapture(self):
        """Display message based on password input"""
        
        self.stopPreview()
        print('Stopped Preview')
        
        print(self.stopEvent.is_set())
        
        #time.sleep(10)
        self.startPreview(self.camera_name.get())
        #print('Yup started preview')
        
        '''
        #im = captureImage()
        img = np.array([[[0,0,0]]]) #1-pixel image
        img = Image.fromarray(img, 'RGB')
        i = 0
        while os.path.exists(self.saveDir.get() + '%03d.jpeg' % i):
            i += 1
        filename = self.saveDir.get() + '%03d.jpeg' % i
        img.save(filename) #this depends on what I use to get the image
        '''
        
    def startPreview(self, camera_number):
        '''start camera preview'''
        #print(self.stopEvent.is_set())
        self.stopEvent = None
        self.thread = None
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.stopEvent = threading.Event()
        
        self.cam = cv2.VideoCapture(camera_number)
        
        self.thread.start()
        print('Preview started')

    def stopPreview(self):
        
        try:
            self.stopEvent.set()
            print('Set thread stop signal')
            self.previewPanel.configure(image = None)
            print('Removed preview image')
            self.cam.release()
            print('Released camera')
        except AttributeError:
            print('Camera not connected')
       
    def videoLoop(self):
        
        try:
            stopCheck = self.stopEvent.is_set()
            if not stopCheck:
                # Capture frame-by-frame
                r,frame = self.cam.read()
                
                if r:
                    width = 400
                    frame_sized = cv2.resize(frame, (width, np.int(9*width/16)))  
                
                    # Display the resulting frame
                    #cv2.imshow(WINDOW_NAME,frame_sized)
                
                    frame_sized = cv2.cvtColor(frame_sized,cv2.COLOR_BGR2RGBA)
                    image = Image.fromarray(frame_sized)
                    img = ImageTk.PhotoImage(image)
                    
                    self.previewPanel.configure(image = img)
                    self.previewPanel.image = img 
                
                    self.previewPanel.after(10, self.videoLoop())
                    
        except AssertionError:
            print('Some dumb AssertionError')
            
        except RuntimeError:
            print('WTF is a runtime error?')
            raise
        
        except:
            print('Probably some stupid cv2 error... threading... yada, yada, yada')
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
