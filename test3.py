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
#Field_sites,Independent_bands =0,0
# ask_yes_no.py
def writetofile(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    #convert to pandas
    df=DataFrame(list1)
    #write to csv
    df.to_csv(path+file1,index=False,header=False)

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

'''
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
#
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
import statsmodels.api as sm
def spatial_error_xy(table1): #images of the same dataset should take the same spatial error
    datasets=np.array((np.array(table1)[:,6]),dtype=float)
    sp_errors=list(datasets)
    for data1 in list(set(datasets)):
        print ('data1',data1)
        expected_spatial_error=float(list(filter(lambda x: float(x[6])==data1, table1))[0][2])
        e=np.random.normal(0, expected_spatial_error, 1)[0]
        sp_errors=list(map(lambda x:x if x!=data1 else e,sp_errors))
        #replace sp_errors dataset with a random error
    return sp_errors
def spatial_error_t(table1): #images of the same dataset should take the same spatial error
    datasets=np.array((np.array(table1)[:,6]),dtype=float)
    sp_errors=list(datasets)
    for data1 in list(set(datasets)):
        print ('data1',data1)
        expected_orientation_error=float(list(filter(lambda x: float(x[6])==data1, table1))[0][4])
        e=np.random.normal(0, expected_orientation_error, 1)[0]
        t_errors=list(map(lambda x:x if x!=data1 else e,sp_errors))
        #replace sp_errors dataset with a random error
    return t_errors
#i=-1
def add_error(table1):#
    images=[]    
    sp_errorsX=spatial_error_xy(table1)
    sp_errorsY=spatial_error_xy(table1)
    sp_errorsT=spatial_error_t(table1)
    L=-1
    for row1 in table1:
        [path1,file1,expected_spatial_error,expected_spectral_error,expected_orientation_error,BAND_number,dataset_number,resolution,is_Spatial,is_Spectral]=row1
        print (path1,file1)
        L+=1
        #read the image
        img=np.array(img2pandas(path1,file1).values.tolist())
        if is_Spectral==1:
            #add the spectral error
            img[:,2]= img[:,2]+ np.random.normal(0, expected_spectral_error, len(img))
        if is_Spatial==1:
            #add the spatial error
            img[:,0]= img[:,0]+ sp_errorsX[L]
            img[:,1]= img[:,1]+ sp_errorsY[L]
            img_Cx,img_Cy=(min(img[:,0])+max(img[:,0]))/2,(min(img[:,1])+max(img[:,1]))/2
            img=((img-([img_Cx,img_Cy,0]))*(frrot3(np.array([0,0,3.141592653589793/180*sp_errorsT[L]]))[0]))+([img_Cx,img_Cy,0])
        images.append(img)
    return images,sp_errorsX,sp_errorsY,sp_errorsT
def creat_table4corlation(rss,fields,RStable):#creat the table by adding the first RS band
    Biomass=[]
    for f1 in fields:
        Biomass.extend(f1)
    bs=np.array((np.array(RStable)[:,5:7]),dtype=float)
    rs_bands=[]
    for f1 in set(tuple(i) for i in bs):#list(set(bands)):
        #extend all inputs of same dataset and same band 
        rs_band=[]
        j=-1
        for i in bs:
            j+=1
            if i[0]==f1[0] and i[1]==f1[1]:
                print(i)
                rs_band.extend(rss[j])
        rs_band=list(map(lambda x: list(x),rs_band))
        rs_bands.append(rs_band)
    Biomass=list(map(lambda x: list(x),Biomass))
    return Biomass,rs_bands

def regression_images(list1,pixelWidth,pixelHeight,list2):# list1 is the lowest resolution band 
    # filter the list with -1 values
    #list1=filter(lambda a: a[2] != -1, list1)
    list2=filter(lambda a: a[2] != -1, list2)
    column1=[]
    # for each pixle 
    for pixel1 in list1:
        #filter it spatially 
        x,y,v=pixel1
        pixel2=filter(lambda a: a[0] >x-abs(pixelWidth)/2 and a[1] >y-abs(pixelHeight)/2  and a[0] <x+abs(pixelWidth)/2 and a[1] <y+abs(pixelHeight)/2  , list2)
        # find the sum of biomass
        try: 
            v2=sum(np.array(pixel2)[:,-1])
            # extend the list
            column1.append([v2])
            #print('ok')
        except:
            column1.append([np.nan])#print('empty')
    #bioRS=filter(lambda a: a[1] != -1, bioRS) 
    return column1
def run_iterations(field_data,RStable,equation_degree,MC_iteration):
    #arrange rs by resolution
    RStable=sorted(RStable,key=lambda l:-float(l[7]))
    Lowest_resolution=float(RStable[0][7])
    #max_res=max(np.array(np.array(RStable)[:,7], dtype=float))
    #indesx_max_res=list(np.array(np.array(RStable)[:,7], dtype=float)).index(max_res)
    record=[]
    F_spatial_errors_Dx=[]
    F_spatial_errors_Dy=[]
    F_spatial_errors_Dt=[]
    RS_spatial_errors_Dx=[]
    RS_spatial_errors_Dy=[]
    RS_spatial_errors_Dt=[]
    for iteration1 in range(MC_iteration):
        #i+=1
        #add spectral
        fields,Fsp_errorsX,Fsp_errorsY,Fsp_errorsT=add_error(field_data)# adds errors to the field data 
        F_spatial_errors_Dx.append(Fsp_errorsX)
        F_spatial_errors_Dy.append(Fsp_errorsY)
        F_spatial_errors_Dt.append(Fsp_errorsT)
        rss,Rsp_errorsX,Rsp_errorsY,Rsp_errorsT=add_error(RStable)# adds errors to the rss data 
        RS_spatial_errors_Dx.append(Rsp_errorsX)
        RS_spatial_errors_Dy.append(Rsp_errorsY)
        RS_spatial_errors_Dt.append(Rsp_errorsT)
        Biomass,rs_bands=creat_table4corlation(rss,fields,RStable)# combines field data #combines any similar band and dataset RS inputs because they are subsetted from the same origional raster
        #now we can start corelation
        #filter list1 if at all needed
        list1=list(map(lambda x: list(x),rss[0]))
        #creat the table
        final_table=[regression_images(list1,Lowest_resolution,Lowest_resolution,fields)]
        for j in rss:
            X=regression_images(list1,Lowest_resolution,Lowest_resolution,list(j))
            for i in range(equation_degree):
                print(i+1)
                final_table.append(list(map(lambda x:x[0]**i+1, X)))#int(s) if s.isdigit() else 0
        final_table=pd.DataFrame(np.transpose(np.array(final_table[1:])))
        final_table.dropna()
        final_table=final_table.values.tolist()
        xx=list(map(lambda x:x[1:], final_table))
        yy=list(map(lambda x:x[0], final_table))
        model = sm.OLS(yy,xx)#len(final_table[1:].values.tolist())==len(final_table[0].values.tolist())
        results = model.fit()
        print('Parameters: ', results.params,'R2: ', results.rsquared)
        record.append([i,[results.params],results.rsquared])
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','record1.csv',record,3)
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dx.csv',RS_spatial_errors_Dx,1)
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dy.csv',RS_spatial_errors_Dy,1)
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dt.csv',RS_spatial_errors_Dt,1)
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dx.csv',F_spatial_errors_Dx,1)
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dy.csv',F_spatial_errors_Dy,1)
    writetofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dt.csv',F_spatial_errors_Dt,1)
# a function to add a list to an existed external file 
def Addtofile(path,file1,list1,NoOfColumns=3):
    try:
        os.stat(path)
    except:
        os.makedirs(path)
    F=path+file1
    text1=''
    for i in range(NoOfColumns):
        text1=text1+',x'+str(i+1)
    text1=text1[1:]
    read=open(F,'a')
    if NoOfColumns!=1:
        for line in list1:
            exec(text1+'= [float(value) for value in line]')
            for i in range(NoOfColumns):
                exec("read.write(str(x"+str(i+1)+'))')
                read.write(',')
            read.write('\n')
    else:
        for line in list1:
            exec("read.write(str(line))")
            read.write('\n')
    read=0
'''
def iteration1(table1):#tables=[field_data,RStable]
    [field_data,RStable]=table1
    fields,Fsp_errorsX,Fsp_errorsY,Fsp_errorsT=add_error(field_data)# adds errors to the field data 
    rss,Rsp_errorsX,Rsp_errorsY,Rsp_errorsT=add_error(RStable)# adds errors to the rss data 
    Biomass,rs_bands=creat_table4corlation(rss,fields,RStable)# combines field data #combines any similar band and dataset RS inputs because they are subsetted from the same origional raster
    #now we can start corelation
    #filter list1 if at all needed
    list1=list(map(lambda x: list(x),rss[0]))
    #creat the table
    final_table=[regression_images(list1,Lowest_resolution,Lowest_resolution,fields)]
    for j in rss:
        X=regression_images(list1,Lowest_resolution,Lowest_resolution,list(j))
        for i in range(1):
            print(i+1)
            final_table.append(list(map(lambda x:x[0]**i+1, X)))#int(s) if s.isdigit() else 0
    final_table=pd.DataFrame(np.transpose(np.array(final_table[1:])))
    final_table.dropna()
    final_table=final_table.values.tolist()
    xx=list(map(lambda x:x[1:], final_table))
    yy=list(map(lambda x:x[0], final_table))
    model = sm.OLS(yy,xx)#len(final_table[1:].values.tolist())==len(final_table[0].values.tolist())
    results = model.fit()
    print('Parameters: ', results.params,'R2: ', results.rsquared)
    record.append([i,[results.params],results.rsquared])
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','record1.csv',record,3)
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dx.csv',Rsp_errorsX,1)
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dy.csv',Rsp_errorsY,1)
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','RS_spatial_errors_Dt.csv',Rsp_errorsT,1)
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dx.csv',Fsp_errorsX,1)
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dy.csv',Fsp_errorsY,1)
    addtofile('C:/Users/ahalboabidallah/Desktop/mont_carlo/','F_spatial_errors_Dt.csv',Fsp_errorsT,1)
    #arrange rs by resolution
RStable=sorted(RStable,key=lambda l:-float(l[7]))
Lowest_resolution=float(RStable[0][7])
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

tables=[[field_data,RStable]]*MC_iteration
p=Pool()
p.map(iteration1,tables)
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