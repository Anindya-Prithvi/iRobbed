import socket
import numpy as np

class SimpleReceiver_Sahas:
    def __init__(self, port=12469, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        self.s.listen(5)
        self.conn, self.addr = self.s.accept()
        self.meshnum = 1
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
        header = self.conn.recv(64)
        if len(header) == 0:
            return None, None
        # parse header
        size = self.parse_header(header)
        # receive data
        data = self.recvall(self.conn, size)
        # parse data
        self.parse_data(data, type)
        return data, frame, type

    def parse_header(self, header):
        size = int(header[:64].decode('utf-8').strip())
        return size

    def parse_data(self, data):
        with open(f"savdir/{self.meshnum}") as f:
            f.write(data)
            self.meshnum+=1

    def close(self):
        self.conn.close()
        self.s.close()

if __name__ == "__main__":
    while True:
        receiver = SimpleReceiver_Sahas()
        while True:
            data, frame,type = receiver.receive()
            if data is None:
                break
            print(frame, data.shape, type)
        receiver.close()