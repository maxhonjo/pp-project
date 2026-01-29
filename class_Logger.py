"""
loggerClass.py
Contains the Logger class.
"""
from pynput import keyboard
from datetime import datetime

class Logger:

    """init -> only creates self.log
    """
    def __init__(self):

        self.log = []

    """on_press function for listener
    """
    def on_press(self, key):

        strokeData = (str(key), datetime.now())
        self.log.append(strokeData)

        print(f'added {strokeData} to log')

    """on_release function for listener
    """
    def on_release(self, key):

        if key == keyboard.Key.esc:
            return False

    """call .start() to start logging
    """
    def start(self):

        with keyboard.Listener(
            on_press = self.on_press,
            on_release = self.on_release) as listener:
            
            listener.join()

    """call .stop() to stop logging
    """
    def stop(self):

        keyboard.Listener.stop()