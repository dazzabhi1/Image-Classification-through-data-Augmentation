from tkinter import *
from PIL import Image, ImageTk
import tkinter.ttk as ttk
from tkinter import filedialog

LARGE_FONT=("Verdana",8)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                   
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)
        
        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)

        edit = Menu(menu)
        edit.add_command(label="Show Img", command=self.showImg)
        edit.add_command(label="Show Text", command=self.showText)
        menu.add_cascade(label="Edit", menu=edit)

        #lbl1 = Label(self,textvariable=folder_path)
        #lbl1.place(x=15,y=71)
        label=Label(text="Choose file for train dataset",font=LARGE_FONT)
        label.place(x=15,y=10)
        global folder_path
        folder_path = StringVar()
        b_button = ttk.Button(text="Browse", command=browse_button)
        b_button.place(x=15,y=31)
        lbl1 = Label(self,textvariable=folder_path)
        lbl1.place(x=15,y=53)
        pr = ttk.Button(text="Click to view progress", command=self.progress)
        pr.place(x=15, y=92)

    def showImg(self):
        load = Image.open("panda.jpg")
        render = ImageTk.PhotoImage(load)

        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

    def showText(self):
        text = Label(self, text="Hey there good lookin!")
        text.pack()
        
    def client_exit(self):
        exit()
    
    def progress(self):
        #ft = ttk.Frame()
        self.pack(expand=True, fill=BOTH, side=TOP)
        self.place(x=15,y=125)
        pb_hd = ttk.Progressbar(self, orient='horizontal', mode='determinate')
        pb_hd.pack(expand=False, fill=BOTH, side=TOP)
        #pb_hd.place(x=15,y=10)
        pb_hd.start(150)
        #browseButton.place(x=10, y=20)
def browse_button():
        filename = filedialog.askdirectory()
        folder_path.set(filename)
        print(filename)
        path='Path: '+filename
        print(path)
        

root = Tk()
root.geometry("400x300")
app = Window(root)
root.mainloop()  