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
from distutils.core import setup
from Cython.Build import cythonize
setup(
  name=,
  ext_modles=cythonize(""))

#Field_sites,Independent_bands =0,0
# ask_yes_no.py

"""
def readtopandas(path1,file1,alradyhasheader=0):#
    #F='C:/Users/ahalboabidallah/Desktop/test.csv'
    F=path1+file1
    #add header to the file if there is no header #
    if alradyhasheader==0:
        #generate a header 
        df = pd.read_csv(F,header=None)
    else:
        df = pd.read_csv(F)#needs a csv with a header line
    return df   

def readtolist(path,file1,NoOfColumns=3,alradyhasheader=0):
    df=readtopandas(path,file1,alradyhasheader=0)
    list1=df.values.tolist()
    return list1

def img2pandas(path1,file1):
    #open file
    src_dataset = gdal.Open(path1+'/'+file1)
    z = src_dataset.ReadAsArray()
    #read georeferencing
    (xmin,res1,tilt1,ymin,tilt2,res2)=src_dataset.GetGeoTransform()
    ys,xs=np.shape(z)
    x = np.array([list(np.linspace(xmin, xmin+(xs-1)*res1, xs))]*(ys))
    y = np.transpose(np.array([list(np.linspace(ymin, ymin+(ys-1)*res1, ys))]*(xs)))
    #z1=list(z.ravel())
    #y1=list(y.ravel())
    #1=list(x.ravel())
    data=np.array([list(x.ravel()),list(y.ravel()),list(z.ravel())])
    #'C:/Users/ahalboabidallah/Desktop/ash_farm_new/profiles/profiles/results/AllGround.tiff'
    return pd.DataFrame(data,index=['X','Y','Z']).transpose()

def regression_images(list1,pixelWidth,pixelHeight,list2):# list1 is the independent with lower resolution and list2=biomass with higher resolution 
    # filter the list with -1 values
    list1=filter(lambda a: a[2] != -1, list1)
    list2=filter(lambda a: a[2] != -1, list2)
    bioRS=[]
    # for each pixle 
    for pixel1 in list1:
        #filter the biomass spatially 
        x,y,v=pixel1
        biomass=filter(lambda a: a[0] >x-abs(pixelWidth)/2 and a[1] >y-abs(pixelHeight)/2  and a[0] <x+abs(pixelWidth)/2 and a[1] <y+abs(pixelHeight)/2  , list2)
        # find the sum of biomass
        try: 
            v2=sum(np.array(biomass)[:,-1])
            # extend the list
            bioRS.append([v,v2])
            #print('ok')
        except:
            pass#print('empty')
    #bioRS=filter(lambda a: a[1] != -1, bioRS) 
    return bioRS
import numpy
def frrot3(theta,U0=numpy.matrix(numpy.identity(3))):
    ct = np.mat([[math.cos(theta[0])],[math.cos(theta[1])],[math.cos(theta[2])]])
    st = np.mat([[math.sin(theta[0])],[math.sin(theta[1])],[math.sin(theta[2])]])
    if max((theta.shape)) > 0:
        R1 = np.mat([[1,0,0],[0,ct[0],-st[0]],[0,st[0],ct[0]]])
        R = R1;
    
    if max((theta.shape)) > 1:
        R2 = np.mat([[float(ct[1]),0,-st[1]],[0,1,0],[st[1],0,ct[1]]]);
        R = R2*R;
    
    if max((theta.shape)) > 2:
        R3 = np.mat([[float(ct[2]),-st[2],0],[st[2],ct[2],0],[0,0,1]]);
        R = R3*R;
    
    R = R*U0;
    #checked and vetted
    return (R,R1,R2,R3)


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
if answer == 'yes':
    print('Yes!')
    MyFrame().mainloop()
    first1().mainloop( )
    
else:  # 'no'
    print('No!')
    first().mainloop( )
    table1=[[]]*Field_sites

#------------------------------------'''
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
        self.l3=Label(text="# MC_iteration")
        self.l3.grid(row=2)
        self.e3 = Entry()
        self.e3.insert(END, 'C:/field.csv')
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
root = Tk()
root.withdraw() # won't need this
answer2 = messagebox.askquestion('Save '+title1+' inputs table', 'Would you like to save '+title1+' inputs table??')
root.deiconify
root.destroy()
if answer2=='yes':
     SaveTable().mainloop()
field_data=table1
#table1=[6,7,8,9]
#'''
title1='Remote Sensing'
#root = Tk()
#root.withdraw() # won't need this
#answer3 = messagebox.askquestion('start', 'Do you have an excel file of the Remote Sensing inputs?')
#root.deiconify
#root.destroy()
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
root = Tk()
root.withdraw() # won't need this
answer2 = messagebox.askquestion('Save '+title1+' inputs table', 'Would you like to save '+title1+' inputs table??')
root.deiconify
root.destroy()
if answer2=='yes':
     SaveTable().mainloop()
RStable=table1
import ogr, gdal, osr, os
import numpy as np
import itertools
from math import sqrt,ceil

"""
#i=-1
for iteration1 in range(MC_iteration):
    #i+=1
    #add spectral
    fields=[]
    for field in field_data:
        [path1,file1,expected_spatial_error,expected_spectral_error,expected_orientation_error,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral]=field
        #read the image
        img=np.array(img2pandas(path1,file1).values.tolist())
        if is_Spectral==1:
            #add the spectral error
            img[:,2]= img[:,2]+ np.random.normal(0, expected_spectral_error, len(img))
        if is_Spatial==1:
            #add the spatial error
            img[:,0]= img[:,0]+ np.random.normal(0, expected_spatial_error, 1)
            img[:,1]= img[:,1]+ np.random.normal(0, expected_spatial_error, 1)
        fields.append(img)
    rss=[]
    for rs in RStable:
        [path,file,expected_spatial_error,expected_spectral_error,expected_orientation_error,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral]=rs
        #read the image
        img=np.array(img2pandas(path1,file1).values.tolist())
        if is_Spectral==1:
            #add the spectral error
            img[:,2]= img[:,2]+ np.random.normal(0, expected_spectral_error, len(img))
        if is_Spatial==1:
            #add the spatial error
            img[:,0]= img[:,0]+ np.random.normal(0, expected_spatial_error, 1)
            img[:,1]= img[:,1]+ np.random.normal(0, expected_spatial_error, 1)
        rss.append(img)
    # make regression 
    
    # save resulted regression
# read the independents and apply the error equation  
#a=[]
#for i in range(1000):
#    a.append(np.random.normal(0, Espatial, 1)[0])

