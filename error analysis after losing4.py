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
if __name__ == '__main__':
    from FsError import *
if __name__ == '__main__':
    root = Tk()
    root.withdraw() # won't need this
    answer = messagebox.askquestion('start', 'Do you have an excel files of the fieldwork and RS inputs?')
    root.deiconify
    root.destroy()
    title1='Fieldwork'
class first(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("inputs")
        self.l1=Label(text="# Field_sites")
        self.l1.grid(row=0)
        self.l2=Label(text="# Independent_bands")
        self.l2.grid(row=1)
        self.l3=Label(text="# MC_iteration")
        self.l3.grid(row=2)
        self.e1 = Entry()
        self.e2 = Entry()
        self.e3 = Entry()
        self.e1.insert(END, '10')
        self.e2.insert(END, '5')
        self.e3.insert(END, '1000')
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.ok = Button(text="Ok", command=self.first_ok)
        self.ok.grid(row=3)
        #self.Button(first, text='Quit', command=first.destroy).grid(row=4, column=0, sticky=W, pady=4)
    def first_ok(self):
        print(" Field_sites: %s\n Independent_bands: %s\n Independent_bands: %s\n" % (self.e1.get(), self.e2.get(),self.e3.get()))
        global Field_sites
        global Independent_bands
        global MC_iteration
        Field_sites=int(self.e1.get())
        Independent_bands =int(self.e2.get())
        MC_iteration=int(self.e3.get())
        self.master.destroy()

class first1(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("inputs")
        self.l3=Label(text="# MC_iteration")
        self.l3.grid(row=2)
        self.e3 = Entry()
        self.e3.insert(END, '1000')
        self.e3.grid(row=2, column=1)
        self.ok = Button(text="Ok", command=self.first_ok)
        self.ok.grid(row=3)
        #self.Button(first, text='Quit', command=first.destroy).grid(row=4, column=0, sticky=W, pady=4)
    def first_ok(self):
        print(self.e3.get())
        global Field_sites
        global Independent_bands
        global MC_iteration
        MC_iteration=int(self.e3.get())
        self.master.destroy()

class MyFrame(Frame):
    #global fname
    global title1
    def __init__(self):
        Frame.__init__(self)
        self.master.title('browse for '+title1+' file')
        self.master.config(height=10)
        self.master.rowconfigure(2, weight=1,pad=10)
        self.master.columnconfigure(3, weight=3,pad=10)
        self.grid()
        self.entry = Entry()
        self.entry.grid(row=1,column=1,sticky='we',columnspan=25)
        self.ok = Button(text="Ok", command=self.close_window1)
        self.ok.grid(row=2, column=1)
        self.ok.config( height = 1, width = 5 )
        self.browse = Button(text="Browse", command=self.load_file)
        self.browse.grid(row=2, column=2)
        self.browse.config( height = 1, width = 5 )
    def close_window1(self): 
        global fname
        global table1
        fname= self.entry.get()
        print(fname)
        table1=readtopandas(fname,'',alradyhasheader=1).values.tolist()
        self.master.destroy()
        return
    def load_file(self):
        fname = askopenfilename(filetypes=(("csv", "*.csv"),
                                           ("All files", "*.*") ))
        if fname:
            try:
                self.entry.delete(0, END)
                self.entry.insert(0, fname)
                
            except:  
                showerror("Open Source File", "Failed to read file\n'%s'" % fname)
            return
    def close_window(self): 
        self.master.destroy()
        return
if __name__ == '__main__':
    if answer == 'yes':
        print('Yes!')
        MyFrame().mainloop()
        first1().mainloop( )
    else:  # 'no'
        print('No!')
        first().mainloop( )
        table1=[[]]*Field_sites

#------------------------------------
class Application(Frame):
    global table1
    global answer
    global title1
    def __init__(self,master):
        Frame.__init__(self, master)
        self.master = master
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
        self.cloneButton = Button (self.frameZero,text='add data ▼',bg = "gold", command=self.clone)
        self.cloneButton.grid(row=0)
        self.OkButton = Button (self.frameTwo,text='OK ✓',bg = "green", command=self.Done)
        self.OkButton.grid(row=0,column=1)
        self.OkButton = Button (self.frameTwo,text='colour map', command=self.change_colour)
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
    root = Tk()
    app = Application(root)
    app.master.title("Fieldwork Data")
    app.mainloop()
    
class SaveTable(Frame):
    global table1
    def __init__(self):
        Frame.__init__(self)
        self.master.title("inputs")
        self.l3=Label(text="#Save to:")
        self.l3.grid(row=2)
        self.e3 = Entry()
        self.e3.insert(END, 'C:/Users/ahalboabidallah/Desktop/mont_carlo/field.csv')
        self.e3.grid(row=2, column=1)
        self.br = Button(text="Browse", command=self.file_save)
        self.br.grid(row=2,column=2)
        self.ok = Button(text="Ok", command=self.first_ok)
        self.ok.grid(row=3)
        
        #self.Button(first, text='Quit', command=first.destroy).grid(row=4, column=0, sticky=W, pady=4)
    def first_ok(self):
        print(self.e3.get())
        df=DataFrame(table1,columns=['path','file','expected_spatial_error','expected_spectral_error','expected_orientation_error','BAND_number','dataset_number','resolution','is_Spatial','is_Spectral'])
        df.to_csv(self.e3.get(),index=False,index_col = False)
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
    if answer == 'yes':
        print('Yes!')
        MyFrame().mainloop()
        #first1().mainloop( )
    else:  # 'no'
        print('No!')
        #first().mainloop( )
        table1=[[]]*Independent_bands
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
    global tables
    tables=[[field_data,RStable]]*MC_iteration
    global Lowest_resolution
#max_res=max(np.array(np.array(RStable)[:,7], dtype=float))
#indesx_max_res=list(np.array(np.array(RStable)[:,7], dtype=float)).index(max_res)
#record=[]
#F_spatial_errors_Dx=[]
#F_spatial_errors_Dy=[]
#F_spatial_errors_Dt=[]
#RS_spatial_errors_Dx=[]
#RS_spatial_errors_Dy=[]
#RS_spatial_errors_Dt=[]

        #add spectral
#print('tables[0]',tables[0])

if __name__ == '__main__':
    p=Pool()
    p.map(FsError.iteration1,tables)
    p.close()
    p.join()
    #
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