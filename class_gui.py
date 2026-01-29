import tkinter as tk
from class_Logger import Logger
from threading import Thread

class GUI:

    def __init__(self):
        
        self.root = tk.Tk()
        self.root.geometry("300x300+50+50")

        self.myLogger = Logger()
        
        self.startButton = tk.Button(self.root,
                                     text = "Start Button",
                                     command = self.startLogging
                                     )
        self.startButton.pack()


        self.seeLogButton = tk.Button(self.root,
                                text = "See Log",
                                command = self.printLog
                                )
        
        self.seeLogButton.pack()


    def startLogging(self):
        
        print("started logging")

        new_thread = Thread(target=self.myLogger.start, daemon=True)
        new_thread.start()

    def printLog(self):

        for key, time in self.myLogger.log:
            print(key, time)




    def run(self):

        self.root.mainloop()