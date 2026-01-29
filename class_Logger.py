"""
loggerClass.py



"""



from pynput import keyboard
from datetime import datetime



    
    
class Logger:

    def __init__(self):

        self.log = []


    def on_press(self, key):

        strokeData = (str(key), datetime.now())
        self.log.append(strokeData)

        print(f'added {strokeData} to log')


    def on_release(self, key):

        if key == keyboard.Key.esc:
            return False


    def start(self):

        with keyboard.Listener(
            on_press = self.on_press,
            on_release = self.on_release) as listener:
            
            listener.join()

    def stop(self):

        keyboard.Listener.stop()