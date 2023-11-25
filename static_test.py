import socket
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
















def thread_post(data, frame, type):
    print('[socket] Posting...',frame,data.shape)
    #Post data to server
    # data is always np array
    # frame is always int
    # determine size of data
    
    # data serialized
    data = data.tobytes()

    # data size
    size = len(data)

    # create a header of 64 bytes and set it to the size of the data and frame number
    header = f"{size:<64}{frame:<63}{type:<1}".encode('utf-8')

    # create a socket
    sock.sendall(header)
    sock.sendall(data)

