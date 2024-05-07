from tkinter import *
#from PIL import ImageTk, Image
from tkinter import ttk


FLAG=False
def mainloop():
    global FLAG
    FLAG=True
    win = Tk()

# Define the geometry of the window
    win.geometry("1024x660")



    flag = False
    def text_label():
        global flag 
        flag = True
        Label(win, text= "Woohoo! Let's begin!", font= ('Helvetica 10 bold')).pack(pady=20)

        print(flag)

   #Configure the Button to trigger a new event
        button.configure(command= close_win)
        FLAG=True

#Define a function to close the event
    def close_win():
        win.destroy()
    button= Button(win, text= "Start the Museum Tour",height=6, width=20, bg='white', 
                    activebackground='red',font= ('Helvetica 10 bold'), command= text_label)
    button.pack(side=TOP,expand=1)
    win.mainloop()
    print(flag)
    if flag==True:
        FLAG=True