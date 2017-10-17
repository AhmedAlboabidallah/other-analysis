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
from FsError import *

class first(Frame):
    def __init__(self):
        self.root =Tk()
        self.root.iconbitmap(r'C:\Users\ahalboabidallah\Documents\ico.ico')
        self.root.title("Inputs")
        Frame.__init__(self)
        #self.img =PhotoImage(file = r'C:\Users\ahalboabidallah\Documents\ico.png')
        #self.root.tk.call('wm', 'iconphoto', self.root._w, self.img)
        self.FieldFile=LabelFrame(text="Field Data Table File--------------",bg='lemon chiffon',height=150,width=500)
        self.FieldFile.grid(row=0, column=0,sticky=N+E+W+S)
        
        self.isFieldFile=StringVar()
        self.yesFieldFile=Checkbutton(self.FieldFile, text='Import a table file', variable=self.isFieldFile,command=self.enable,bg='lemon chiffon')
        self.yesFieldFile.grid(sticky=W)
        self.yesFieldFile.select()
        
        self.FF=LabelFrame(self.FieldFile,text="Table file",bg='lemon chiffon')
        self.FF.grid(row=1,column=0)
        self.browse=  Button(self.FF,text="Browse",command=self.load_filef)
        self.browse.grid(row=1, column=3,sticky=W+E)
        self.File_f=Entry(self.FF)
        self.File_f.grid(row=1, column=0,sticky=W+E)
        self.number = StringVar()
        self.No=LabelFrame(self.FieldFile,text="Number of inputs",bg='lemon chiffon')
        self.numberChosen = ttk.Combobox(self.No, width=12, textvariable=self.number, state="readonly")
        self.numberChosen.grid(column=1, row=1, sticky=E+W)
        self.numberChosen["values"] = list(range(100))#(1, 2, 3, ..., 100)
        self.numberChosen.current(1)
        
        self.ok = Button(text="OK ✓",width=7, command=self.first_ok,bg='lightgreen')
        self.ok.grid(row=100,column=0,columnspan=4,sticky=N,padx=10, pady=10)
        
        self.RSFile=LabelFrame(text="Remote Sensing Data Table File",bg='lavender',height=150)
        self.RSFile.grid(row=0, column=2,sticky=N+E+W+S)
        self.isRSFile=StringVar()
        self.yesRSFile=Checkbutton(self.RSFile, text='Import a table file', variable=self.isRSFile,command=self.enable2,bg='lavender')
        self.yesRSFile.grid(sticky=W)
        self.yesRSFile.select()
        self.RSF=LabelFrame(self.RSFile,text="Table file",bg='lavender')
        self.RSF.grid(row=10,column=0,columnspan=3)
        self.browseRS=  Button(self.RSF,text="Browse",command=self.load_file2)
        self.browseRS.grid(row=1, column=3,sticky=W+E)
        self.File2=Entry(self.RSF)
        self.File2.grid(row=1, column=0,columnspan=2,sticky=W+E)
        self.number2 = StringVar()
        self.RSNo=LabelFrame(self.RSFile,text="Number of inputs",bg='lavender')
        self.RSnumberChosen = ttk.Combobox(self.RSNo, width=12, textvariable=self.number2, state="readonly")
        self.RSnumberChosen.grid(column=1, row=1, sticky=E+W)
        self.RSnumberChosen["values"] = list(range(100))#(1, 2, 3, ..., 100)
        self.RSnumberChosen.current(3)
        
        self.MC=LabelFrame(text="Monte Carlo Iteration",bg='azure',height=150)
        self.MC.grid(row=1, column=0, columnspan=2,sticky=N+E+W+S)
        #self.l=Label(self.MC,text=' ')
        #self.l.grid(row=0, column=0,columnspan=2,sticky=W+E)
        self.e3=Entry(self.MC)
        self.e3.grid(row=1, column=0,columnspan=2,sticky=W+E)
        self.e3.insert(0, '1000')
        
        self.ProcessingType=LabelFrame(bg="bisque",text='Processing Type:')
        self.ProcessingType.grid(row=1,column=2,sticky=N+E+W+S)
        self.processingVar=StringVar()
        #self.l_ProcessingType= Label(self.ProcessingType, text='Processing Type:___________')
        #self.l_ProcessingType.grid(row=0,column=5,sticky=W+E)
        self.PB_ProcessingType= Radiobutton(self.ProcessingType, text='Pixel Based',value='1',variable=self.processingVar,anchor=W, bg="bisque",command=self.processing_type)
        self.OB_ProcessingType= Radiobutton(self.ProcessingType, text='Object Based',value='0',variable=self.processingVar,anchor=W, bg="bisque",command=self.processing_type)
        self.PB_ProcessingType.grid(row=1,column=5,sticky=W+E+N+S)
        self.OB_ProcessingType.grid(row=2,column=5,sticky=W+E+N+S)
        self.PB_ProcessingType.invoke()
        self.brs_ProcessingType=LabelFrame(self.ProcessingType,bg="bisque",text ='Segmentation File:')
        #self.File1=   Label(self.brs_ProcessingType,text='Segmentation File:')
        #self.File1.grid(row=2, column=5,sticky=W)
        self.FileP=Entry(self.brs_ProcessingType)
        self.FileP.grid(row=3, column=5,sticky=W)
        #self.File1.insert(0,'0')
        self.browse=  Button(self.brs_ProcessingType,text="Browse",command=self.load_fileP)
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
    def first_ok(self):
        global Field_sites
        global Independent_bands
        global MC_iteration
        global answerF
        global answerRS
        global Field_file
        global RS_file
        MC_iteration=int(self.e3.get())
        Field_sites=int(self.numberChosen.get())
        Independent_bands=int(self.RSnumberChosen.get())
        self.master.destroy()    
        answerF=['no','yes'][int(self.isFieldFile.get())]
        answerRS=['no','yes'][int(self.isRSFile.get())]
        print('Field_sites;',Field_sites,'Independent_bands:',Independent_bands,'MC_iteration:',MC_iteration)
        print('answerF;',answerF,'answerRS:',answerRS)
        if answerF=='yes':
            Field_file=self.Ffile
        if answerRS=='yes':
            RS_file=self.RSfile
        global ProcessingType
        ProcessingType=['Object','Pixel'][int(self.processingVar.get())]
        print("ProcessingType: ",ProcessingType)
        global Processing
        Processing=[ProcessingType]
        if self.processingVar.get()=='0':
            global SegmentationFile
            SegmentationFile=self.FileP.get()
            print("SegmentationFile: ",SegmentationFile)
            Processing.extend(SegmentationFile)
    def enable2(self):
        if self.isRSFile.get()=='1':
            self.RSF.isgridded=True
            self.RSF.grid(row=1,column=0,columnspan=3)
            self.RSNo.isgridded=False
            self.RSNo.grid_forget()
            print('browse')
        else:
            self.RSF.isgridded=False
            self.RSF.grid_forget()
            self.RSNo.isgridded=True
            self.RSNo.grid(row=1,column=0,columnspan=3)
            
    def enable(self):
        if self.isFieldFile.get()=='1':
            self.FF.isgridded=True
            self.FF.grid(row=1,column=0,columnspan=3)
            self.No.isgridded=False
            self.No.grid_forget()
            print('browse')
        else:
            self.FF.isgridded=False
            self.FF.grid_forget()
            self.No.isgridded=True
            self.No.grid(row=1,column=0,columnspan=3)
        #
    def load_file2(self):
        self.RSfile = askopenfilename(filetypes=(("csv", "*.csv"),("All files", "*.*") ))
        self.File2.delete(0, END)
        self.File2.insert(0, self.RSfile)      
    def load_fileP(self):
        fname = askopenfilename(filetypes=(("shp", "*.shp"),
                                           ("All files", "*.*") ))
        if fname:
            try:
                self.FileP.delete(0, END)
                self.FileP.insert(0, fname)
            except:  
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return
    def load_filef(self):
        self.Ffile = askopenfilename(filetypes=(("csv", "*.csv"),("All files", "*.*") ))
        self.File_f.delete(0, END)
        self.File_f.insert(0, self.Ffile)

if __name__ == '__main__':
    first().mainloop( )


    
#----------------------
class Application(Frame):
    global table1
    global answer
    global title1
    
    def __init__(self,master):
        Frame.__init__(self, master)
        self.master = master
        self.master.iconbitmap(r'C:\Users\ahalboabidallah\Documents\ico.ico')
        self.master.minsize(width=500, height=200)
        self.initUI()
        self.fname=''
        #self.kk=0
        self.colour_list=readtolist('C:/Users/ahalboabidallah/Desktop/','col.csv')
    def initUI(self):
        #def __init__(self, master=None):
        self.colour_list=[]
        self.kk=0
        self.master.title(title1)
        self.frameOne = Frame(self.master)
        self.frameOne.grid(row=1,column=0)
        self.frameTwo = Frame(self.master)
        self.frameTwo.grid(row=2, column=0)
        self.frameZero = Frame(self.master)
        self.frameZero.grid(row=0, column=0)
        global table1
        #self.table1=table1
        self.lableset=['path','file','expected_spatial_error','expected_spectral_error','expected_orientation_error','BAND_number','dataset_number','resolution']
        self.k=-1
        for col in self.lableset:
            self.k+=1
            Label(self.frameOne,text=col).grid(row=0, column=self.k)
        Frame.__init__(self)
        #self.number = 1
        self.widgets = []
        self.grid()
        #self.createWidgets()
        self.table1=table1
        self.this_entry=[]
        self.all_entries=[]
        for j in self.table1:
            self.clone()
        #def createWidgets(self):
        # = Frame(self)
        self.cloneButton = Button (self.frameZero,text='Add Data ▼',bg = "gold", command=self.clone)
        self.cloneButton.grid(row=0)
        self.OkButton = Button (self.frameTwo,text='OK ✓',bg = "lightgreen", command=self.Done)
        self.OkButton.grid(row=0,column=1)
        self.OkButton = Button (self.frameTwo,text='Show Colour-code', command=self.change_colour)
        self.OkButton.grid(row=0, column=0)
    def clone(self):
        self.kk+=1
        print('kk',self.kk)
        if answer=='no' or self.kk>len(self.table1):
            self.fname = askopenfilename(filetypes=(("tif", "*.tif"),
                                           ("All files", "*.*") ))
        else:
            try:
                self.fname=table1[self.kk-1][0]+table1[self.kk-1][1]
            except:
                self.fname=''
        self.this_entry=[]
        for i in range(7):
            exec('self.var'+str(self.kk)+'_'+str(i)+'=Variable(self)')
            try:
                exec('self.var'+str(self.kk)+'_'+str(i)+'.set(table1[self.kk-1][i])')
            except:
                exec('self.var'+str(self.kk)+'_'+str(i)+'.set(1)')
            exec('self.en'+str(self.kk)+'_'+str(i)+' = Entry(self.frameOne,textvariable=self.var'+str(self.kk)+'_'+str(i)+')')
            exec('self.en'+str(self.kk)+'_'+str(i)+'.grid(row=self.kk+1,column=i)')
            #print('creat entry','self.en'+str(self.kk)+'_'+str(i))
        exec('self.en'+str(self.kk)+'_'+str(0)+'.delete(0,END)')
        exec('self.en'+str(self.kk)+'_'+str(0)+'.insert(0,str(self.fname[0:-len(os.path.basename(self.fname))]))')
        #self.this_entry.append(self.en01)
        #self.en02 = Entry(self.frameOne)
        #self.en02.grid(row=self.number,column=1)
        exec('self.en'+str(self.kk)+'_'+str(1)+'.delete(0,END)')
        exec('self.en'+str(self.kk)+'_'+str(1)+'.insert(0,str(os.path.basename(self.fname)))')
        #self.this_entry.append(self.en02)
        #self.lll=[,,'','','','','']
        #for i in range(4):
         #   self.en1 = Entry(self.frameOne)
          #  self.en1.grid(row=self.number,column=i+2)
           # self.this_entry.append(self.en1)
        i=1+i
        ds = gdal.Open(self.fname)
        (xmino,res1,tilt1,ymino,tilt2,res2)=ds.GetGeoTransform()
        exec('self.var'+str(self.kk)+'_'+str(i)+'=Variable(self)')
        try:
            exec('self.var'+str(self.kk)+'_'+str(i)+'.set(table1[self.kk-1][i])')
        except:
            exec('self.var'+str(self.kk)+'_'+str(i)+'.set(res1)')
        exec('self.en'+str(self.kk)+'_'+str(i)+' = Entry(self.frameOne,textvariable=self.var'+str(self.kk)+'_'+str(i)+')')
        exec('self.en'+str(self.kk)+'_'+str(i)+'.grid(row=self.kk+1,column=i)')
        i=1+i
        #self.kk=len(self.all_entries)
        #exec("brs"+str(len(self.all_entries))+"=Button (self.frameOne,text='Browse',bg = 'green',textvariable=str(len(self.all_entries)), command=lambda self:load(self))")#
        #exec("brs"+str(len(self.all_entries))+".grid(row=self.number,column=i+1)")
        exec('self.var'+str(self.kk)+'_'+str(i)+'=Variable(self)')
        try:
            exec('self.var'+str(self.kk)+'_'+str(i)+'.set(table1[self.kk-1][i])')
        except:
            exec('self.var'+str(self.kk)+'_'+str(i)+'.set(1)')
        #exec('self.var'+str(self.kk)+'_'+str(i)+'.set(1)')
        #self.var.append(self.var1)
        exec('self.Spatial = Checkbutton(self.frameOne, text="Spatial error",variable=self.var'+str(self.kk)+'_'+str(i)+')')
        exec('self.Spatial.grid(row=self.kk+1,column=i)')
        exec('self.this_entry.append(self.Spatial)')
        i=1+i
        exec('self.var'+str(self.kk)+'_'+str(i)+'=Variable(self)')
        try:
            exec('self.var'+str(self.kk)+'_'+str(i)+'.set(table1[self.kk-1][i])')
        except:
            exec('self.var'+str(self.kk)+'_'+str(i)+'.set(1)')
        #self.var.append(self.var2)
        exec('self.Spectral = Checkbutton(self.frameOne, text="Spectral error",variable=self.var'+str(self.kk)+'_'+str(i)+')')
        exec('self.Spectral.grid(row=self.kk+1,column=i)')
        exec('self.this_entry.append(self.Spectral)')
        #self.all_entries.append(self.this_entry)
        #self.number += 1
      
    def change_colour(self):
        #c = user.get() #Get the entered text of the Entry widget
        for wid in range(self.kk):#
            #
            exec('u=int(self.var'+str(wid+1)+'_'+str(5)+'.get())')
            #exec("print ('u',u)")
            #print('self.en'+str(wid+1)+'_'+str(4)+'.configure(bg =self.colour_list[u])')
            exec('self.en'+str(wid+1)+'_'+str(5)+'.configure(bg =self.colour_list[u])')
            #exec('self.en'+str(wid+1)+'_'+str(5)+'.configure(bg ="green")')
            #
            exec('u=int(self.var'+str(wid+1)+'_'+str(6)+'.get())')
            #exec("print ('u',-u)")
            #print('self.en'+str(wid+1)+'_'+str(6)+'.configure(bg =self.colour_list[-u])')
            exec('self.en'+str(wid+1)+'_'+str(6)+'.configure(bg =self.colour_list[-u])')
            #    pass
    def Done(self):
        #maketable
        #print(self.kk)
        self.table=[]
        for i in range(self.kk):
            self.i_input=[]
            for j in range(10):
                #print('self.i_input.append(self.var'+str(i+1)+'_'+str(j)+'.get())')
                exec('self.i_input.append(self.var'+str(i+1)+'_'+str(j)+'.get())')
                
            self.table.append(self.i_input)
        #global table1
        table1=self.table
        global table1
        self.master.destroy()
        print('self.table',np.array(self.table),table1)
        return        
        #exec()                             
    #def ADDTABLE(self):
        #global table1
#
if __name__ == "__main__":
    title1='Fieldwork'#
    answer=answerF
    if answer == 'yes':
        table1=readtopandas(Field_file,'',alradyhasheader=1).values.tolist()
    else:
        table1=[[]]*Field_sites
    root = Tk()
    app = Application(root)
    app.master.title("Fieldwork Data")
    app.mainloop()
    
class SaveTable(Frame):
    global table1
    def __init__(self):
        self.root =Tk()
        self.root.iconbitmap(r'C:\Users\ahalboabidallah\Documents\ico.ico')
        Frame.__init__(self)
        self.master.title("Save Inputs")
        self.l3=Label(text="Save to:")
        self.l3.grid(row=2)
        self.e3 = Entry()
        self.e3.insert(END, 'C:/Users/ahalboabidallah/Desktop/mont_carlo/field.csv')
        self.e3.grid(row=2, column=1)
        self.br = Button(text="Browse", command=self.file_save)
        self.br.grid(row=2,column=2)
        self.ok = Button(text='OK ✓', command=self.first_ok,bg = 'lightgreen')
        self.ok.grid(row=3,columnspan=2,sticky=N)
        
        #self.Button(first, text='Quit', command=first.destroy).grid(row=4, column=0, sticky=W, pady=4)
    def first_ok(self):
        print(self.e3.get())
        df=DataFrame(table1,columns=['path','file','expected_spatial_error','expected_spectral_error','expected_orientation_error','BAND_number','dataset_number','resolution','is_Spatial','is_Spectral'])
        df.to_csv(self.e3.get(),index=False)
        self.master.destroy()
    def file_save(self):
        f = asksaveasfile(mode='w', defaultextension=".csv")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        #text2save = str(text.get(1.0, END)) # starts from `1.0`, not `0.0`
        self.e3.delete(0,END)
        self.e3.insert(0,f.name)      
#MyFrame().mainloop()
if __name__ == '__main__':
    root = Tk()
    root.withdraw() # won't need this
    answer2 = messagebox.askquestion('Save '+title1+' inputs table', 'Would you like to save '+title1+' inputs table??')
    root.deiconify
    root.destroy()
    if answer2=='yes':
         SaveTable().mainloop()
    field_data=table1
    #table1=[6,7,8,9]
    #
    
    title1='Remote Sensing'
    
#root = Tk()
#root.withdraw() # won't need this
#answer3 = messagebox.askquestion('start', 'Do you have an excel file of the Remote Sensing inputs?')
#root.deiconify
#root.destroy()
if __name__ == '__main__':
    answer=answerRS
    if answer == 'yes':
        table1=readtopandas(RS_file,'',alradyhasheader=1).values.tolist()
    else:
        table1=[[]]*Field_sites
if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    app.master.title(title1+" Data")
    app.mainloop()
if __name__ == '__main__':
    root = Tk()
    root.withdraw() # won't need this
    answer2 = messagebox.askquestion('Save '+title1+' inputs table', 'Would you like to save '+title1+' inputs table??')
    root.deiconify
    root.destroy()
    if answer2=='yes':
         SaveTable().mainloop()
    RStable=table1

if __name__ == '__main__':
    RStable=sorted(RStable,key=lambda l:-float(l[7]))
    global Lowest_resolution
    Lowest_resolution=float(RStable[0][7])
    print('tables are builted')
    #global tables
    #tables=[[field_data,RStable]]*MC_iteration
    #global Lowest_resolution
    
#max_res=max(np.array(np.array(RStable)[:,7], dtype=float))
#indesx_max_res=list(np.array(np.array(RStable)[:,7], dtype=float)).index(max_res)
#record=[]
#F_spatial_errors_Dx=[]
#F_spatial_errors_Dy=[]
#F_spatial_errors_Dt=[]
#RS_spatial_errors_Dx=[]
#RS_spatial_errors_Dy=[]
#RS_spatial_errors_Dt=[]

class Model(Frame):
    def __init__(self):
        self.root =Tk()
        self.root.minsize(width=150, height=150)
        self.root.iconbitmap(r'C:\Users\ahalboabidallah\Documents\ico.ico')
        Frame.__init__(self)
        self.root.title("Model Type")
        self.ModelType=LabelFrame(text="Model Type: ",bg='gainsboro',height=350)
        self.ModelType.grid(row=0,column=0,sticky=N+E+W+S)
        self.CBX1 = ttk.Combobox(self.ModelType,textvariable='Parametric_Regression',state="readonly")
        self.CBX1['values'] = ['Parametric_Regression','Support_Vector_Machine','Neural Network','Gaussian_Process','Random_Forest','K_Nearest_Neighbours']
        self.CBX1.current(0)
        #self.e1.insert(END, '10')
        self.CBX1.grid(row=0, column=1,sticky=E+W,padx=20, pady=20)
        self.CBX1.bind('<<ComboboxSelected>>',self.ChangeInputs)   # add='+'??
        self.c=3
        self.nodes = []
        self.current = StringVar(value='0')
        self.ok = Button(text="OK ✓",width=7, command=self.first_ok,bg='lightgreen')
        self.ok.grid(row=100,column=0, columnspan=5,sticky=N,padx=10, pady=10)
        #----------------------------------------------------------------------
        #gaussian_process http://scikit-learn.org/stable/modules/gaussian_process.html
        self.Gaussian=LabelFrame(self.ModelType,text="Gaussian_Process",bg='lemon chiffon')
        #kernel Radial-basis function (RBF), Matérn, Rational quadratic,Exp-Sine-Squared, Dot-Product, 
        self.GaussianK=LabelFrame(self.Gaussian,text="Kernel Type")
        self.GaussianK.grid(row=0, column=0,sticky=W)
        self.GP_K_CBX = ttk.Combobox(self.GaussianK,textvariable='Kernel Type')
        self.GP_K_CBX['values'] = ['RBF','Matérn','Rational_Quadratic','Exp-Sine-Squared','Dot-Product']
        self.GP_K_CBX.current(0)
        self.GP_K_CBX.grid(row=1, column=0,sticky=W+E)
        self.GaussianNL=LabelFrame(self.Gaussian,text="Noise Level (%)")
        self.GaussianNL.grid(row=2, column=0,sticky=W+E)
        self.Gaussian_NL_Scale=Scale(self.GaussianNL, from_=0, to=100,orient=HORIZONTAL,bg='lightblue',foreground='darkblue')
        self.Gaussian_NL_Scale.grid(row=3, column=0,columnspan=2,sticky=E+W+N)
        self.GaussianNL.grid(row=3, column=0,sticky=W+E)
        #self.ok = Button(self.Gaussian,text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,sticky=N,padx=20, pady=20)
        #----------------------------------------------------------------------
        #random forests
        #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        self.Random_Forest=LabelFrame(self.ModelType,text="Bootstrap")
        #bootstrap ['true', 'False']#
        self.rRFB=StringVar()
        self.Random_Forest_B1=Radiobutton(self.Random_Forest, text='Use bootstrap', value=True,variable=self.rRFB)
        self.Random_Forest_B2=Radiobutton(self.Random_Forest, text='Do not use bootstrap', value=False,variable=self.rRFB)
        self.Random_Forest_B1.grid(sticky=W)
        self.Random_Forest_B2.grid(sticky=W)
        self.Random_Forest_B1.invoke()
        #self.ok = Button(self.Random_Forest,text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,column=1, columnspan=2,sticky=N,padx=20, pady=20)
        #--------------------------------------------------------
        #k nearest Neighbours http://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html 
        #weights=['uniform','distances']
        #k
        self.K_n=LabelFrame(self.ModelType,text="K_Nearest_Neighbours")
        self.K_n_W=LabelFrame(self.K_n,text="Weights Type")
        self.K_n_W.grid()
        self.K_n_K=LabelFrame(self.K_n,text="The k Value")
        self.K_n_K.grid()
        self.kn=StringVar()
        self.K_n_B1=Radiobutton(self.K_n_W, text='Uniform', value=True,variable=self.kn)
        self.K_n_B2=Radiobutton(self.K_n_W, text='Distance Based', value=False,variable=self.kn)
        self.K_n_B1.grid(sticky=W)
        self.K_n_B2.grid(sticky=W)
        self.K_n_B1.invoke()
        self.K_n_k=Entry(self.K_n_K,width=6,justify='center')
        self.K_n_k.insert(END,'10')
        self.K_n_k.grid(row=60,sticky=E+W)
        #self.ok = Button(self.K_n, text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,column=1, columnspan=2,sticky=N,padx=20, pady=20)
        #----------------------------------------------------------------------
        #stepwise regression 
        #http://planspace.org/20150423-forward_selection_with_statsmodels/
        #self.SWregression=LabelFrame(self.ModelType,text="Regression model degree")
        #self.SWregression.grid(row=2,column=0,columnspan=3)
        #self.ok = Button(self.SWregression,text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100)
        #self.ii=0
        #self.SWregressionVar=StringVar()
        #for degree in ['first order','second order','third order']:
        #    self.ii+=1
        #    exec("self.STbutton"+str(self.ii)+" = Radiobutton(self.SWregression, text=degree, value=self.ii,variable=self.SWregressionVar)")
        #    exec("self.STbutton"+str(self.ii)+".grid(row=self.ii+1, column=0, sticky='w')")
        #exec('self.STbutton1.invoke()')
        #self.ok = Button(self.SWregression, text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,column=1, columnspan=2,sticky=N,padx=20, pady=20)
        #----------------------------------------------------------------------
        self.GaussianK=LabelFrame(self.Gaussian,text="Kernel Type")
        self.GaussianK.grid(row=0, column=0,sticky=W)
        self.GP_K_CBX = ttk.Combobox(self.GaussianK,textvariable='Kernel Type')
        self.GP_K_CBX['values'] = ['RBF','Matern','RationalQuadratic','ExpSineSquared','DotProduct','WhiteKernel','ConstantKernel']
        self.GP_K_CBX.current(0)
        self.GP_K_CBX.grid(row=1, column=0,sticky=W+E)
        self.GaussianNL=LabelFrame(self.Gaussian,text="Noise Level (%)")
        self.GaussianNL.grid(row=2, column=0,sticky=W+E)
        self.Gaussian_NL_Scale=Scale(self.GaussianNL, from_=0, to=100,orient=HORIZONTAL,bg='lightblue',foreground='darkblue')
        self.Gaussian_NL_Scale.grid(row=3, column=0,columnspan=2,sticky=E+W+N)
        self.GaussianNL.grid(row=3, column=0,sticky=W+E)
        #self.ok=Button(self.Gaussian,text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,sticky=N,padx=20, pady=20)
        #--------------------------------------------------------standarad reg
        self.regression=LabelFrame(self.ModelType,text="Regression model degree")
        self.regression.grid(row=2,column=0,columnspan=3)
        #
        self.listLayrs=[]
        self.listLayrsl=[]
        self.i=0
        #
        self.regressionVar=StringVar()
        self.processingVar=StringVar()
        #self.ok=Button(self.regression,text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,sticky=N,padx=20, pady=20)
        #
        #self.l_reg.grid(row=1)
        self.button1 = Radiobutton(self.regression, text='first order', value=1,variable=self.regressionVar)
        self.button1.grid(row=2, column=0, sticky='w')
        self.button2 = Radiobutton(self.regression, text='second order', value=2,variable=self.regressionVar)
        self.button2.grid(row=3, column=0, sticky='w')
        self.button3 = Radiobutton(self.regression, text='third order', value=3,variable=self.regressionVar)
        self.button3.grid(row=4, column=0, sticky='w')
        self.button3 = Radiobutton(self.regression, text='multiplicative regression', value=4,variable=self.regressionVar)
        self.button3.grid(row=5, column=0, sticky='w')
        self.button3 = Radiobutton(self.regression, text='exponential regression', value=5,variable=self.regressionVar)
        self.button3.grid(row=6, column=0, sticky='w')
        self.button1.invoke()
        #----------------------------------------------------------------------
        #Support_Vector_Machine http://scikit-learn.org/stable/modules/svm.html   http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        #Linear Kernel, Gaussian RBF kernels, Polynomial Kernel+Degree
        self.svm=LabelFrame(self.ModelType,text="SVM model kernel")
        self.SVM_Var=StringVar()
        self.j=3
        #self.ok=Button(self.svm,text="OK ✓", command=self.first_ok)
        #self.ok.grid(row=100,sticky=N,padx=20, pady=20)
        for kernel in ['Linear', 'Gaussian_RBF', 'Polynomial']:
            self.j+=1
            exec("self.svmbutton"+str(self.j)+" = Radiobutton(self.svm, text=kernel+' Kernel', value=self.j,variable=self.SVM_Var,command=self.kernel)")
            exec("self.svmbutton"+str(self.j)+".grid(row=self.j+1, column=0, sticky='w')")
        exec('self.svmbutton4.invoke()')
        self.svmPD=LabelFrame(self.svm,text="Degree")
        #self.svmPD.grid(row=60)
        self.svmPDegree=Entry(self.svmPD,width=6,justify='center')
        self.svmPDegree.insert(END,'3')
        self.svmPDegree.grid(row=60)
        #------------------------------------- -------------------------------
        #self.Knn=Frame()
        self.Knn=LabelFrame(self.ModelType,text='KNN Spesifications')
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
    def kernel(self):
        if int(self.SVM_Var.get())==6:
            self.svmPD.isgridded=True
            self.svmPD.grid(row=70,column=0,columnspan=3,sticky=EW)#
        else:           
            try:
                self.svmPD.isgridded=False
                self.svmPD.grid_forget()
            except:
                pass
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
        global model_spec
        model_spec=[]
        global Model
        Model=self.CBX1.get()
        model_spec.extend([Model])
        print("Model: ",self.CBX1.get())
        #['Parametric_Regression','Stepwise_Regression','Support_Vector_Machine','Neural Network','Gaussian_Process','Random_Forest','K_Nearest_Neighbours']
        if self.CBX1.get()=='Parametric_Regression':
            print (self.regressionVar.get())
            global RegDegree
            RegDegree=int(self.regressionVar.get())
            print("RegDegree: ",RegDegree)
            model_spec.extend([RegDegree])
        #elif self.CBX1.get()=='Stepwise_Regression':
        #    print (self.SWregressionVar.get())
        #    global RegDegree1
        #    RegDegree1=int(self.SWregressionVar.get())
        #    print("RegDegree: ",RegDegree1)
        elif self.CBX1.get()=='Support_Vector_Machine':
            print (self.SVM_Var.get())
            global SVM_kernel
            SVM_kernel=['linear', 'rbf', 'poly'][int(self.SVM_Var.get())-4]
            print("SVM kernel: ",SVM_kernel)
            model_spec.extend([SVM_kernel])
            global poly_degree
            poly_degree=1
            if SVM_kernel=='Polynomial':
                poly_degree=self.svmPDegree.get()#could be 1 or not 1
                print("Poly_Degree: ",poly_degree)
            model_spec.extend([poly_degree])
        elif self.CBX1.get()=='Neural Network':
            global NumberOfLayers
            global Layers
            NumberOfLayers=int(self.e_knn_layers.get())
            Layers=[]
            for i in range(len(self.listLayrs)):
                Layers.append(int(self.vars[i].get()))
            print("NumberOfLayers: ",NumberOfLayers,"> Layers: ",Layers)
            model_spec.extend([NumberOfLayers,Layers])
        elif self.CBX1.get()=='Gaussian_Process':
            global Gaussian_Process_kernal
            Gaussian_Process_kernal=self.GP_K_CBX.get()
            global Gaussian_noise_level
            Gaussian_noise_level= self.Gaussian_NL_Scale.get()/100
            print('Gaussian_Process_kernal',Gaussian_Process_kernal)
            print('Gaussian_noise_level',int(Gaussian_noise_level*100),'%')
            model_spec.extend([Gaussian_Process_kernal,Gaussian_noise_level])
        elif self.CBX1.get()=='Random_Forest':
            print (self.regressionVar.get())
            global Random_Forest_Bootstrap
            Random_Forest_Bootstrap=[True,False][int(self.rRFB.get())-1]#
            print("Random_Forest_Bootstrap: ",Random_Forest_Bootstrap)
            model_spec.extend([Random_Forest_Bootstrap])
        elif self.CBX1.get()=='K_Nearest_Neighbours':
            #print (self.regressionVar.get())
            global K_Nearest_Neighbours_weights#
            K_Nearest_Neighbours_weights=['uniform','distance'][int(self.kn.get())-1]#
            print("K_Nearest_Neighbours_weights: ",K_Nearest_Neighbours_weights)
            global K_Nearest_Neighbours_k#
            K_Nearest_Neighbours_k=int(self.K_n_k.get())
            print("K_Nearest_Neighbours_k: ",K_Nearest_Neighbours_k)
            model_spec.extend([K_Nearest_Neighbours_weights,K_Nearest_Neighbours_k])
            #
        #global ProcessingType
        #ProcessingType=['Object','Pixel'][int(self.processingVar.get())]
        #print("ProcessingType: ",ProcessingType)
        #if self.processingVar.get()=='0':
        #    global SegmentationFile
        #    SegmentationFile=self.File1.get()
        #    print("SegmentationFile: ",SegmentationFile)
        self.master.destroy()
        #add spectral
    def ChangeInputs(self, event):
        self.CBX1.selection_clear()
        #['Parametric_Regression','Stepwise_Regression','Support_Vector_Machine','Neural Network','Gaussian_Process','Random_Forest','K_Nearest_Neighbours']
        if self.CBX1.get()=='Parametric_Regression':
            self.Knn.isgridded=False
            self.Knn.grid_forget()
            self.svm.isgridded=False
            self.svm.grid_forget()
            self.Gaussian.isgridded=False
            self.Gaussian.grid_forget()
            self.Random_Forest.isgridded=False
            self.Random_Forest.grid_forget()
            self.K_n.isgridded=False
            self.K_n.grid_forget()
        #    self.SWregression.isgridded=False
        #    self.SWregression.grid_forget()
            self.regression.isgridded=True
            self.regression.grid(row=2,column=0,columnspan=3)
        #elif self.CBX1.get()=='Stepwise_Regression':
        #    self.Knn.isgridded=False
        #    self.Knn.grid_forget()
        #    self.svm.isgridded=False
        #    self.svm.grid_forget()
        #    self.Gaussian.isgridded=False
        #    self.Gaussian.grid_forget()
       #     self.regression.isgridded=False
        #    self.regression.grid_forget()
        #    self.K_n.isgridded=False
        #    self.K_n.grid_forget()
        #    self.Random_Forest.isgridded=False
        #    self.Random_Forest.grid_forget()
        #    self.SWregression.isgridded=True
        #    self.SWregression.grid(row=2,column=0,columnspan=3)
        elif self.CBX1.get()=='Support_Vector_Machine':
            self.regression.isgridded=False
            self.regression.grid_forget()
            self.Knn.isgridded=False
            self.Knn.grid_forget()
            self.Gaussian.isgridded=False
            self.Gaussian.grid_forget()
            self.Random_Forest.isgridded=False
            self.Random_Forest.grid_forget()
            self.K_n.isgridded=False
            self.K_n.grid_forget()
       #     self.SWregression.isgridded=False
       #     self.SWregression.grid_forget()
            self.svm.isgridded=True
            self.svm.grid(row=2,column=0,columnspan=3)
        elif self.CBX1.get()=='Neural Network':
            self.regression.isgridded=False
            self.regression.grid_forget()
            self.svm.isgridded=False
            self.svm.grid_forget()
            self.Gaussian.isgridded=False
            self.Gaussian.grid_forget()
            self.Random_Forest.isgridded=False
            self.Random_Forest.grid_forget()
            self.K_n.isgridded=False
            self.K_n.grid_forget()
        #    self.SWregression.isgridded=False
        #    self.SWregression.grid_forget()
            self.Knn.isgridded=True
            self.Knn.grid(row=2,column=0,columnspan=4)
        elif self.CBX1.get()=='Gaussian_Process':
            self.regression.isgridded=False
            self.regression.grid_forget()
            self.Knn.isgridded=False
            self.Knn.grid_forget()
            self.svm.isgridded=False
            self.svm.grid_forget()
            self.Random_Forest.isgridded=False
            self.Random_Forest.grid_forget()
            self.K_n.isgridded=False
            self.K_n.grid_forget()
        #    self.SWregression.isgridded=False
        #    self.SWregression.grid_forget()
            self.Gaussian.isgridded=True
            self.Gaussian.grid(row=2,column=0,columnspan=3)
        elif self.CBX1.get()=='Random_Forest':
            self.Knn.isgridded=False
            self.Knn.grid_forget()
            self.svm.isgridded=False
            self.svm.grid_forget()
            self.Gaussian.isgridded=False
            self.Gaussian.grid_forget()
            self.regression.isgridded=False
            self.regression.grid_forget()
            self.K_n.isgridded=False
            self.K_n.grid_forget()
        #    self.SWregression.isgridded=False
        #    self.SWregression.grid_forget()
            self.Random_Forest.isgridded=True
            self.Random_Forest.grid(row=2,column=0,columnspan=3)
        elif self.CBX1.get()=='K_Nearest_Neighbours':
            self.Knn.isgridded=False
            self.Knn.grid_forget()
            self.svm.isgridded=False
            self.svm.grid_forget()
            self.Gaussian.isgridded=False
            self.Gaussian.grid_forget()
            self.regression.isgridded=False
            self.regression.grid_forget()
            self.Random_Forest.isgridded=False
            self.Random_Forest.grid_forget()
        #    self.SWregression.isgridded=False
        #    self.SWregression.grid_forget()
            self.K_n.isgridded=True
            self.K_n.grid(row=2,column=0,columnspan=3)
        else:
            print ('This model is not avilable yet')
        self.CBX1.selection_clear()
if __name__ == '__main__':
        Model().mainloop( )
    
if __name__ == '__main__':
    i=0
    for fd1 in field_data:
        i+=1
        for rs in RStable:
            FsError.subset_image(extend_image=fd1[0]+fd1[1], inDS=rs[0]+rs[1], outDS=rs[0]+'subset/'+str(i)+'/'+rs[1],tolerance=max([float(fd1[2]),float(rs[2])]))#'''

if __name__ == '__main__':
    global tables
    tables=[[field_data,RStable,Processing,model_spec]]*MC_iteration
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