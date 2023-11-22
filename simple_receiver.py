import socket
import numpy as np

class SimpleReceiver:
    def __init__(self, port=34512, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        self.s.listen(5)
        self.conn, self.addr = self.s.accept()
        print('Connected by', self.addr)
        # self.conn.setblocking(0)

    def recvall(self, sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def receive(self):
        # receive header
        header = self.conn.recv(128)
        if len(header) == 0:
            return None, None
        # parse header
        size, frame = self.parse_header(header)
        # receive data
        data = self.recvall(self.conn, size)
        # parse data
        data = self.parse_data(data)
        return data, frame

    def parse_header(self, header):
        size = int(header[:64].decode('utf-8').strip())
        frame = int(header[64:].decode('utf-8').strip())
        return size, frame

    def parse_data(self, data):
        return np.frombuffer(data, dtype=np.float32)

    def close(self):
        self.conn.close()
        self.s.close()

if __name__ == "__main__":
    receiver = SimpleReceiver()
    while True:
        data, frame = receiver.receive()
        if data is None:
            break 
            # can #goto 51...but that will be the worst case
            # wontfix for now
        print(frame, data.shape)
    receiver.close()