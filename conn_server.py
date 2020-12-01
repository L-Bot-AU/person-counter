import socket
import random
from Crypto.Cipher import AES
import winsound

class ConnServer:
    def __init__(self, SERVER_IP="127.0.0.1", PORT=4485):
        self.SERVER_SOCK = (SERVER_IP, PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.SERVER_SOCK)
        aesenc = AES.new(b"automate_egggggg", AES.MODE_GCM)
        message = self.sock.recv(1024)
        self.sock.send(aesenc.encrypt(message))

    def send(self, msg):
        msg = (msg + "\n").encode("latin-1")
        self.sock.send(msg)

    def add(self, n):
        self.send("+"+str(n))

    def sub(self, n):
        self.send("-"+str(n))


class StubConnServer:
    def __init__(self, SERVER_IP="127.0.0.1", PORT=4485):
        print(f"Connection established with {SERVER_IP}:{PORT}")

    def send(self, msg):
        print(f"Sending: '{msg}'")

    def add(self, n):
        self.send("+"+str(n))
        winsound.Beep(1000, 200)

    def sub(self, n):
        self.send("-"+str(n))
        winsound.Beep(500, 200)

