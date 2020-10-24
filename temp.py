from tkinter import *  
from tkinter import ttk
from PIL import ImageTk,Image  
import tkinter as tk
from tkinter.scrolledtext import *
import tkinter.filedialog
import requests
from PIL import ImageTk,Image 
from bs4 import BeautifulSoup
from urllib.request import urlopen

from Classification_SVM import text_Classification_SVM 
from Classification_RF import text_Classification_RF

root = Tk()  

#canvas = Canvas(root, width = 850, height = 531)  
#canvas.pack() 
root.geometry("850x550")
root.title("Document Classification GUI")

#735
ground = Image.open("ttt.jpg")
backgroundImage = ImageTk.PhotoImage(ground)
label=Label()
label.configure(image=backgroundImage)
label.place(relwidth=1, relheight=1)

label2 = Label(root, text= 'File Processing',padx=5, pady=5)
label2.grid(column=0, row=0)  

def clear_text():
	entry.delete('1.0',END)

def clear_display_result():
	tab1_display.delete('1.0',END)


# Clear Text  with position 1.0
def clear_text_file():
	displayed_file.delete('1.0',END)

# Clear Result of Functions
def clear_text_result():
	tab2_display_text.delete('1.0',END)




# Clear entry widget
def clear_compare_text():
	entry1.delete('1.0',END)

def clear_compare_display_result():
	tab1_display.delete('1.0',END)


# Functions for TAB 2 FILE PROCESSER
# Open File to Read and Process
def openfiles():
   file1 = tkinter.filedialog.askopenfilename(filetypes=(("Text Files",".txt"),("All files","*")))
   read_text = open(file1)
   text = read_text.read()
   displayed_file.insert(tk.END,text)
  
    
def get_text_Classification_SVM ():
	raw_text = displayed_file.get('1.0',tk.END)
	final_text = text_Classification_SVM(raw_text)
	result = '\nClassification using algorithm Linear SVM:{}'.format(final_text)
	tab2_display_text.insert(tk.END,result)
   
def get_text_Classification_RF():
	raw_text = displayed_file.get('1.0',tk.END)
	final_text = text_Classification_RF(raw_text)
	result = '\n Classification using algorithm Random Forest:{}'.format(final_text)
	tab2_display_text.insert(tk.END,result)
    
  
def click_me():
    if(i.get()==1):
       get_text_Classification_SVM()  
       
    else:
       get_text_Classification_RF()       
   
  
displayed_file = ScrolledText(root,height=10)# Initial was Text(tab2)
displayed_file.grid(row=2,column=1, columnspan=3,padx=5,pady=5)

# BUTTONS FOR SECOND TAB/FILE READING TAB
b0=Button(root,text="Open File", width=12,command= openfiles,bg="#b9f6ca")
b0.grid(row=3,column=1,padx=10,pady=10)

b1=Button(root,text="Reset ", width=12,command=clear_text_file,bg="#ffccff")
b1.grid(row=3,column=2,padx=10,pady=10)

i=IntVar()

r1=ttk.Radiobutton(root,text="Linear SVM",variable=i,value=1)
r1.grid(row=5,column=1,padx=10,pady=10)

r2=ttk.Radiobutton(root,text="Random Forest",variable=i,value=2)
r2.grid(row=5,column=2,padx=10,pady=10)

b2=Button(root,text=" Classification", width=12,command=click_me,bg="#66d9ff")
b2.grid(row=5,column=3,padx=10,pady=10)

b3=Button(root,text="Clear Result", width=12,command=clear_text_result)
b3.grid(row=3,column=3,padx=10,pady=10)





#listedata.place(x=640, y=37)


tab2_display_text = ScrolledText(root,height=10)
tab2_display_text.grid(row=7,column=1, columnspan=3,padx=5,pady=5)

root.mainloop() 

