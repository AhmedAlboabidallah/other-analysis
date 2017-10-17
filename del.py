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
        self.kkk=0
        self.master.title(title1)
        self.frameOne = Frame(self.master)
        self.frameOne.grid(row=1,column=0)
        self.frameTwo = Frame(self.master)
        self.frameTwo.grid(row=2, column=0)
        self.frameZero = Frame(self.master)
        self.frameZero.grid(row=0, column=0)
        global table1
        #self.table1=table1
        #self.lableset=['path','file','expected_spatial_error','expected_spectral_error','expected_orientation_error','BAND_number','dataset_number','resolution']
        self.lableset=['path','file','BAND_number','dataset_number','resolution']
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
        for i in range(4):
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
            exec('u=int(self.var'+str(wid+1)+'_'+str(2)+'.get())')
            #exec("print ('u',u)")
            #print('self.en'+str(wid+1)+'_'+str(4)+'.configure(bg =self.colour_list[u])')
            exec('self.en'+str(wid+1)+'_'+str(2)+'.configure(bg =self.colour_list[u])')
            #exec('self.en'+str(wid+1)+'_'+str(5)+'.configure(bg ="green")')
            #
            exec('u=int(self.var'+str(wid+1)+'_'+str(3)+'.get())')
            #exec("print ('u',-u)")
            #print('self.en'+str(wid+1)+'_'+str(6)+'.configure(bg =self.colour_list[-u])')
            exec('self.en'+str(wid+1)+'_'+str(3)+'.configure(bg =self.colour_list[-u])')
            #    pass
    def Done(self):
        #maketable
        #print(self.kk)
        self.table=[]
        for i in range(self.kk):
            self.i_input=[]
            for j in range(7):
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
    
##########################################################################################################################################################################
class AddErrors(Frame):
    global table1
    global table1b
    global answer
    global title1
    
    def __init__(self,master):
        Frame.__init__(self, master)
        self.master = master
        self.master.iconbitmap(r'C:\Users\ahalboabidallah\Documents\ico.ico')
        self.master.minsize(width=700, height=200)
        self.initUI()
        self.fname=''
        #self.kk=0
        self.colour_list=readtolist('C:/Users/ahalboabidallah/Desktop/','col.csv')
        self.ErrorMap=[[]]*1000
        #print(self.ErrorMap)
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
        #self.lableset=['path','file','expected_spatial_error','expected_spectral_error','expected_orientation_error','BAND_number','dataset_number','resolution']
        self.lableset=['path','file','expected_spatial_error','expected_orientation_error','expected_spectral_error', 'is it map based?']
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
        for i in table1:
            self.kk+=1
            self.addrow()
            #print('kk',self.kk)
            self.this_entry=[]
        self.OkButton = Button (self.frameTwo,text='OK ✓',bg = "lightgreen", command=self.Done)
        self.OkButton.grid(row=1000,column=1)
    def addrow(self):
        print('kk', self.kk)
        #file path#file name#spatial shift error#spatial rotation error#spectral error
        for i in range(2):
            exec('self.var'+str(self.kk)+'_'+str(i)+'=Variable(self)')
            try:
                exec('self.var'+str(self.kk)+'_'+str(i)+'.set(table1[self.kk-1][i])')
            except:
                exec('self.var'+str(self.kk)+'_'+str(i)+'.set(1)')
            exec('self.en'+str(self.kk)+'_'+str(i)+' = Entry(self.frameOne,textvariable=self.var'+str(self.kk)+'_'+str(i)+')')
            exec('self.en'+str(self.kk)+'_'+str(i)+'.grid(row=self.kk+1,column=i)')
            #print('self.var'+str(self.kk)+'_'+str(i))
        for j in range(3):
            i=j+2
            k=j+7
            exec('self.var'+str(self.kk)+'_'+str(k)+'=Variable(self)')
            try:
                exec('self.var'+str(self.kk)+'_'+str(k)+'.set(table1[self.kk-1][k])')
            except:
                exec('self.var'+str(self.kk)+'_'+str(k)+'.set(1)')
            #print('self.var'+str(self.kk)+'_'+str(k))
            exec('self.en'+str(self.kk)+'_'+str(k)+' = Entry(self.frameOne,textvariable=self.var'+str(self.kk)+'_'+str(k)+')')
            exec('self.en'+str(self.kk)+'_'+str(k)+'.grid(row=self.kk+1,column=i)')
            print('self.en'+str(self.kk)+'_'+str(k)+'.grid(row=self.kk+1,column=i)')
        #is map?
        i=1+i
        k+=1
        #print('i_is',i)
        #print('k_is',k)
        exec('self.var'+str(self.kk)+'_'+str(k)+'=Variable(self)')
        try:
            exec('self.var'+str(self.kk)+'_'+str(k)+'.set(table1[self.kk-1][i])')
        except:
            exec('self.var'+str(self.kk)+'_'+str(k)+'.set(1)')
        print('self.var'+str(self.kk)+'_'+str(k))
        #ismap self.var.append(self.var2)
        exec('self.ismapCB = Checkbutton(self.frameOne, text="Map based",variable=self.var'+str(self.kk)+'_'+str(k)+',command=self.hide_button('+str(self.kk)+'))')
        exec('self.ismapCB.grid(row=self.kk+1,column=i)')
        exec('self.this_entry.append(self.ismapCB)')
        #browes 
        i+=1
        exec("self.BrowseButton"+str(self.kk)+"= Button(self.frameOne,text='Browse',bg = 'gold', command=lambda self=self: self.mapfile("+str(self.kk)+"))")#command = lambda i=i, j=j: update_binary_text(i, j)
        exec("self.BrowseButton"+str(self.kk)+".grid(row="+str(self.kk+1)+", column=i)")
    def hide_button(self,kk):
        print('self.this_entry',self.this_entry)
    #    try:
    #            exec('self.BrowseButton.grid()')
    #    except:
    #            exec('self.BrowseButton.grid_remove()()')
    def Done(self):
        #maketable
        #print(self.kk)
        self.table=[]
        for i in range(self.kk):
            self.i_input=[]
            for jj in range(4):
                j=jj+7
                #print('self.i_input.append(self.var'+str(i+1)+'_'+str(j)+'.get())')
                exec('self.i_input.append(self.var'+str(i+1)+'_'+str(j)+'.get())')
                
            self.table.append(self.i_input)
        #global table1
        table1b=self.table
        global table1
        global TableSave
        self.master.destroy()
        #print('self.table',np.array(self.table),table1)
        TableSave=DataFrame(table1, columns=['path','file','band','dataset','resolution','isSpaitial','isSpectral'])
        TableSave['patial_error']=np.array(table1b)[:,0]
        TableSave['oriontation_error']=np.array(table1b)[:,1]
        TableSave['spectral_error']=np.array(table1b)[:,2]
        TableSave['is_map']=np.array(table1b)[:,3]
        table1=TableSave.values.tolist()
        return 
    def mapfile(self,kk):
        #kk=self.kkk
        nfile= askopenfilename(filetypes=(("tif", "*.tif"),
                                           ("All files", "*.*") ))
        #exec('self.var'+str(kk)+'_9.set("'+nfile+'")')                              
        exec('self.en'+str(kk)+'_9.delete(0, END)')
        exec('self.en'+str(kk)+'_9.insert(0, nfile)')
        #exec()             
if __name__ == "__main__":
    title1='Fieldwork'#
    answer=answerF
    root = Tk()
    app = AddErrors(root)
    app.master.title("Fieldwork Data")
    app.mainloop()  
#
#########################################################################################################################################################################