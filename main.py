from class_Logger import Logger

myLogger = Logger()

myLogger.start()

for key, time in myLogger.log:
    print(key, time)