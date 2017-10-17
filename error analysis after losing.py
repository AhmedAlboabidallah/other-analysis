from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from tkinter import messagebox
#Field_sites,Independent_bands =0,0
# ask_yes_no.py
'''
root = Tk()
root.withdraw() # won't need this
answer = messagebox.askquestion('start', 'Do you have an excel file of the inputs?')
root.deiconify
root.destroy()


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
        Field_sites,Independent_bands =int(self.e1.get()), int(self.e2.get())
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
    def __init__(self):
        Frame.__init__(self)
        self.master.title("browse for file")
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
        fname= self.entry.get()
        print(fname)
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
    
def addBox():
        frame = Frame(root)
        frame.grid()
        #Label(frame, text='From').grid(row=0, column=0)
        ent1 = Entry(frame)
        ent1.grid(row=1, column=0)
        #Label(frame, text='To').grid(row=0, column=1)
        ent2 = Entry(frame)
        ent2.grid(row=1, column=1)
        all_entries.append( (ent1, ent2) )

#------------------------------------'''

from tkinter import *

class Application(Frame):
    global table1
    def __init__(self, master=None):
        global table1
        #self.table1=table1
        self.lableset=['path','file','expected_spatial_error','expected_spectral_error','BAND_number','dataset_number']
        self.k=-1
        for col in self.lableset:
            self.k+=1
            Label(text=col).grid(row=2, column=self.k)
        Frame.__init__(self, master)
        self.number = 5
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
        self.cloneButton = Button (text='add data',bg = "gold", command=self.clone)
        self.cloneButton.grid(row=0)
    def clone(self):
        self.this_entry=[]
        for i in range(6):
            en1 = Entry()
            en1.grid(row=self.number,column=i)
            self.this_entry.append(en1)
        brs=Button (text='Browse',bg = "green")#, command=self.clone)
        brs.grid(row=self.number,column=i+1)
        Spatial = Checkbutton( text="Spatial error")
        Spatial.grid(row=self.number,column=i+2)
        self.this_entry.append(Spatial)
        Spectral = Checkbutton( text="Spectral error")
        Spectral.grid(row=self.number,column=i+3)
        self.this_entry.append(Spectral)
        self.all_entries.append(self.this_entry)
        self.number += 1
    #def ADDTABLE(self):
        #global table1
    
    #ADDTABLE(self)
table1=[1,2,3,4,5]
if __name__ == "__main__":
    app = Application()
    app.master.title("Sample application")
    app.mainloop()