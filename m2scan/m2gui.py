# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:50:02 2014

@author: cpkmanchee
"""

import os
import numpy as np

from tkinter import *
from tkinter.filedialog import asksaveasfilename
from PIL import Image


class Application(Frame):
    """ GUI app to aid in the collection of beam profile data"""
    
    def __init__(self, master):
        """Initialize frame"""
        Frame.__init__(self,master)
        self.grid()
        self.create_widgets()
        
    def create_widgets(self):
        """Create button, text, entry widgets"""
        
        self.camSelectLabel = Label(self, text = 'Select camera:')
        self.camSelectLabel.grid(row = 0, column = 0, columnspan = 1, sticky = W)
        
        camera = StringVar()
        camera.set('--Select--')
        
        camera_list = self.getCameras()
        
        self.camSelect = OptionMenu(self, camera, *camera_list)
        self.camSelect.grid(row = 0, column = 1, columnspan = 1, sticky = W)        
        
        self.cameraLabel = Label(self, textvariable = camera)
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
        
        self.text = Text(self, width = 35, height = 5, wrap = WORD)
        self.text.grid(row=3, column=0, columnspan = 2, sticky = W)
        
    def onCapture(self):
        """Display message based on password input"""
        
        #im = captureImage()
        img = np.array([[[0,0,0]]]) #1-pixel image
        img = Image.fromarray(img, 'RGB')
        i = 0
        while os.path.exists(self.saveDir.get() + '%03d.jpeg' % i):
            i += 1
        filename = self.saveDir.get() + '%03d.jpeg' % i
        img.save(filename) #this depends on what I use to get the image
        
    def getCameras(self):
        '''get list of available cameras'''
        camera_list = ['cam 0', 'cam 1']
        return camera_list
        
    def browseDir(self):
        '''select save directory'''
        directory = asksaveasfilename()
        self.saveDir.set(directory)
        
        
root = Tk()
root.title("Beam Profile Collection")
root.geometry('400x300')

app = Application(root)

root.mainloop()
