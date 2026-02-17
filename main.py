# standard modules
import pynput
from pynput.keyboard import Listener
import tkinter as tk

# custom modules
from gui import GUI



keystrokes = []
gui = GUI(keystrokes=keystrokes)


def on_press(key):

    keystrokes.append(str(key))

def on_release(key):

    pass




with Listener(on_press=on_press, on_release=on_release) as listener:
    gui.run()

    listener.stop()
    listener.join()

