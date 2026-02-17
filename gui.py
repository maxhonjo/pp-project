"""
Docstring for gui
"""
import tkinter as tk

class GUI:

    def __init__(self, keystrokes):

        self.keystrokes = keystrokes

        self.root = tk.Tk()
        self.root.geometry('500x500+100+100')
        self.root.title('Keylogger Project')

        self.root.attributes('-topmost', 1)
        self.root.attributes('-topmost', 0)
        self.root.focus_force()

        self.root.bind('<Escape>', lambda e: self.root.destroy())

        self.create_widgets()


    def create_widgets(self):

        self.frame_all = tk.Frame(self.root)
        self.frame_all.pack(fill='both', expand=True)

        GRID_ROWS = 5
        for i in range(0, GRID_ROWS):
            self.frame_all.grid_rowconfigure(i, weight=1)
        
        GRID_COLUMNS = 5
        for i in range(0, GRID_COLUMNS):
            self.frame_all.grid_columnconfigure(i, weight=1)

        self.title = tk.Label(self.frame_all, text='title goes here', bg='black')
        self.title.grid(row=0, column=0, columnspan=5, sticky='nsew')

        self.button1 = tk.Button(self.frame_all, text='button1')
        self.button1.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.button2 = tk.Button(self.frame_all, text='button2')
        self.button2.grid(row=2, column=0, columnspan=3, sticky='nsew')
        self.button3 = tk.Button(self.frame_all, text='button3')
        self.button3.grid(row=3, column=0, columnspan=3, sticky='nsew')








    def print_keystrokes(self):
        print(self.keystrokes)

    def stop(self):
        self.root.destroy()

    
    def run(self):

        self.root.mainloop()