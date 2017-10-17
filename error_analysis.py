from __future__ import division
from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import showerror
from tkinter import messagebox
from tkinter import ttk
import os
import numpy as np
import pandas as pd
from pandas import *
import gdal
from gdal import *
import cython
from cython.parallel import *
from multiprocessing import Pool
import math
import numpy
import ogr, gdal, osr, os
import numpy as np
import itertools
from math import sqrt,ceil
#from statsmodels.formula.api import ols
import statsmodels.api
#import statsmodels.api as sm
import FsError
import statsmodels as sm
import statsmodels.robust
#from FsError.py import *
#Field_sites,Independent_bands =0,0
# ask_yes_no.py
"""

"""

class Model(Frame):
    def __init__(self):
        self.root =Tk()
        self.root.minsize(width=200, height=75)
        self.root.iconbitmap(r'C:\Users\ahalboabidallah\Documents\ico.ico')
        Frame.__init__(self)
        self.root.title("Model Type and Processing Type")
        self.l1=Label(text="Model Type: ")
        self.l1.grid(row=0)
        self.CBX1 = ttk.Combobox(textvariable='Regression')
        self.CBX1['values'] = ['Regression', 'Neural Network']
        self.CBX1.current(0)
        #self.e1.insert(END, '10')
        self.CBX1.grid(row=0, column=1,sticky=W)
        self.CBX1.bind('<<ComboboxSelected>>',self.ChangeInputs)   # add='+'??
        self.c=3
        self.nodes = []
        self.current = StringVar(value='0')
        #---------------------------------------------------------------------
        self.regression=LabelFrame(text="Regression model degree")
        self.regression.grid(row=2,column=0,columnspan=3)
        #
        self.listLayrs=[]
        self.listLayrsl=[]
        self.i=0
        #
        self.regressionVar=StringVar()
        self.processingVar=StringVar()
        self.ok = Button(text="OK ✓", command=self.first_ok)
        self.ok.grid(row=100)
        
        #self.l_reg.grid(row=1)
        for degree in ['first order','second order','third order']:
            self.i+=1
            exec("self.button"+str(self.i)+" = Radiobutton(self.regression, text=degree, value=self.i,variable=self.regressionVar)")
            exec("self.button"+str(self.i)+".grid(row=self.i+1, column=0, sticky='w')")
        exec('self.button1.invoke()')
        #----------------------------------------------------------------------SVM 
        #Linear Kernel, Gaussian RBF kernels, Polynomial Kernel
        #SVM 
        #Linear Kernel, Gaussian RBF kernels, Polynomial Kernel+Degree
        self.svm=LabelFrame(text="SVM model kernel")
        #self.svm.grid(row=2,column=0,columnspan=3)
        #
        self.SVM_Var=StringVar()
        self.j=3
        self.ok = Button(self.svm,text="OK ✓", command=self.first_ok)
        for kernel in ['Linear', 'Gaussian_RBF', 'Polynomial']:
            self.j+=1
            exec("self.svmbutton"+str(self.j)+" = Radiobutton(self.svm, text=kernel+' Kernel', value=self.j,variable=self.SVM_Var)")
            exec("self.svmbutton"+str(self.j)+".grid(row=self.j+1, column=0, sticky='w')")
        exec('self.svmbutton4.invoke()')
        #------------------------------------- -------------------------------
        #self.Knn=Frame()
              
        self.Knn=LabelFrame(text='KNN Spesifications')
        self.ok = Button(self.Knn,text="OK ✓", command=self.first_ok)
        self.ok.grid(row=100,column=1, columnspan=2,sticky=N)
        
        #self.l_knn=Label(self.Knn,text=        "____________KNN Spesifications____________")
        #self.l_knn.grid(row=2,columnspan=4)
        self.nodesF=LabelFrame(self.Knn,text="#Nodes in each Layer")
        self.nodesF.grid(row=4,columnspan=4, sticky=N+E+W+S)
        self.l_knn_layers=Label(self.Knn,text="#Hidden layers:")
        self.l_knn_layers.grid(row=3,column=0,sticky=W+E)
        
        self.vcmd = (self.Knn.register(self.validate),'%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.e_knn_layers=Entry(self.Knn,width=6,justify='center')
        #self.e_knn_layers=Entry(self.Knn,validate = 'key', validatecommand = self.vcmd)
        self.e_knn_layers.insert(END, str(self.c))
        
        self.e_knn_layers.grid(row=3,column=2)
        self.b_knn_layersUp=Button(self.Knn,text="+", command=self.Up)
        self.b_knn_layersDown=Button(self.Knn,text="-", command=self.Down)
        self.b_knn_layersUp.grid(row=3,column=3,sticky=W)
        self.b_knn_layersDown.grid(row=3,column=1,sticky=E)
        self.verify()
        
        self.ProcessingType=LabelFrame(bg='lemon chiffon',text='Processing Type:')
        self.ProcessingType.grid(row=1,rowspan=6,column=4,sticky=N+E+W+S)
        
        #self.l_ProcessingType= Label(self.ProcessingType, text='Processing Type:___________')
        #self.l_ProcessingType.grid(row=0,column=5,sticky=W+E)
        
        self.PB_ProcessingType= Radiobutton(self.ProcessingType, text='Pixel Based',value='1',variable=self.processingVar,anchor=W, bg='lemon chiffon',command=self.processing_type)
        self.OB_ProcessingType= Radiobutton(self.ProcessingType, text='Object Based',value='0',variable=self.processingVar,anchor=W, bg='lemon chiffon',command=self.processing_type)
        self.PB_ProcessingType.grid(row=1,column=5,sticky=W+E)
        self.OB_ProcessingType.grid(row=2,column=5,sticky=W+E)
        self.PB_ProcessingType.invoke()
        
        self.brs_ProcessingType=LabelFrame(self.ProcessingType,bg='lemon chiffon',text ='Segmentation File:')
        
        #self.File1=   Label(self.brs_ProcessingType,text='Segmentation File:')
        #self.File1.grid(row=2, column=5,sticky=W)
        self.File1=   Entry(self.brs_ProcessingType)
        self.File1.grid(row=3, column=5,sticky=W)
        self.browse=  Button(self.brs_ProcessingType,text="Browse",command=self.load_file)
        self.browse.grid(row=3, column=6,sticky=W+E)
        
        
    def processing_type(self):
        print(self.processingVar.get())
        if self.processingVar.get()=='0':
            self.brs_ProcessingType.isgridded=True
            self.brs_ProcessingType.grid(row=3,column=5,sticky=N+E+W+S)
            print('object')
        else:
            try:
                self.brs_ProcessingType.isgridded=False
                self.brs_ProcessingType.grid_forget()
            except:
                pass
            print('pixel')
    def load_file(self):
        fname = askopenfilename(filetypes=(("shp", "*.shp"),
                                           ("All files", "*.*") ))
        if fname:
            try:
                self.File1.delete(0, END)
                self.File1.insert(0, fname)
            except:  
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return
    
    def Up(self):
        self.c+=1
        self.e_knn_layers.delete(0, "end")
        print(self.c)
        for digit in str(self.c):
            self.e_knn_layers.insert(END,digit)
        self.verify()
        
    def Down(self):
        self.c+=-1
        self.e_knn_layers.delete(0, "end")
        print(self.c)
        for digit in str(self.c):
            self.e_knn_layers.insert(END,digit)
        self.verify()
        
    def validate(self, action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if text in 'qwertyuiopasdfghjklzxcvbnm':
            return False
        else:
            return True
    #self.Up1.create()
    def verify(self):
        for j in self.listLayrs:
            exec(j+'.grid_remove()')
        for j in self.listLayrsl:
            exec(j+'.grid_remove()')
        self.listLayrs=[]
        self.listLayrsl=[]
        self.vars=[]
        for i in range(int(self.e_knn_layers.get())):
            self.vars.append(StringVar(value='4'))
            
            self.listLayrs.append("self.nodes"+str(i))
            self.listLayrsl.append("self.l_knn_layer_"+str(i))
            
            exec("self.l_knn_layer_"+str(i)+"=Label(self.nodesF,text='layer No." +str(i+1)+"')")
            exec("self.l_knn_layer_"+str(i)+".grid(row="+str(5+i)+",column=0,sticky=W+E)")
            
            exec("self.nodes"+str(i)+"= OptionMenu(self.nodesF,self.vars[i], *[str(i) for i in range(100)])")
            exec("self.nodes"+str(i)+".grid(row="+str(5+i)+",column=2)")
        
        #self.Button(first, text='Quit', command=first.destroy).grid(row=4, column=0, sticky=W, pady=4)
    def first_ok(self):
        global Model
        Model=self.CBX1.get()
        print("Model: ",self.CBX1.get())
        
        if self.CBX1.get()=='Regression':
            print (self.regressionVar.get())
            global RegDegree
            RegDegree=int(self.regressionVar.get())
            print("RegDegree: ",RegDegree)
        elif self.CBX1.get()=='Neural Network':
            global NumberOfLayers
            global Layers
            NumberOfLayers=int(self.e_knn_layers.get())
            Layers=[]
            for i in range(len(self.listLayrs)):
                Layers.append(int(self.vars[i].get()))
            print("NumberOfLayers: ",NumberOfLayers,"> Layers: ",Layers)
        global ProcessingType
        ProcessingType=['Object','Pixel'][int(self.processingVar.get())]
        print("ProcessingType: ",ProcessingType)
        if self.processingVar.get()=='0':
            global SegmentationFile
            SegmentationFile=self.File1.get()
            print("SegmentationFile: ",SegmentationFile)
        self.master.destroy()
        #add spectral
        
    def ChangeInputs(self, event):
        if self.CBX1.get()=='Regression':
            self.Knn.isgridded=False
            self.Knn.grid_forget()
            self.regression.isgridded=True
            self.regression.grid(row=2,column=0,columnspan=3)
            print('Regression')
        elif self.CBX1.get()=='Neural Network':
            self.regression.isgridded=False
            self.regression.grid_forget()
            self.Knn.isgridded=True
            self.Knn.grid(row=2,column=0,columnspan=4)
            print('Neural Network')
        else:
            print ('This model is not avilable here yet')
if __name__ == '__main__':
        Model().mainloop( )


'''
if __name__ == '__main__':
    p=Pool()
    p.map(FsError.iteration1,tables)
    p.close()
    p.join()
    #'''
# apply error to RS data
#1 dx, dy and dtheta

# creat regression list
# regret, add parameters and correlation coofisionts.

                
        
        
        # make regression 
        
    # save resulted regression
# read the independents and apply the error equation  
#a=[]
#for i in range(1000):
#    a.append(np.random.normal(0, Espatial, 1)[0])

#'''