import socket
import random
#from Crypto.Cipher import AES
import threading
import winsound

SERVER_IP = "192.168.137.1"
JNRCONNECT_PORT = 9482
SNRCONNECT_PORT = 11498

class ConnServer:
    def __init__(self, SERVER_IP=SERVER_IP, PORT=SNRCONNECT_PORT):
        print("Connecting")
        self.SERVER_SOCK = (SERVER_IP, PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.SERVER_SOCK)
        print("Connected")
        
        print("Authing")
        #self.aesdec = AES.new(b"automate_egggggg", AES.MODE_ECB)
        message = self.sock.recv(16)
        self.sock.send(b"automate_egggggg") #self.aesdec.decrypt(message))
        print("Authed")

    def send(self, msg):
        try:
            self.sock.send(msg.encode("latin-1"))
        except (ConnectionResetError, ConnectionAbortedError):
            print("Lost connection :'(")
            self.__init__(*self.SERVER_SOCK)
            self.send(msg)
    
    def add(self, n : int):
        print(f"Sending +{n}")
        self.send("+"+str(n))

    def sub(self, n : int):
        print(f"Sending -{n}")
        self.send("-"+str(n))


class StubConnServer:
    def __init__(self, SERVER_IP=SERVER_IP, PORT=SNRCONNECT_PORT):
        print(f"Connection established with {SERVER_IP}:{PORT}")
        self.total = 0
        
    def send(self, msg):
        #print(f"Sending: '{msg}'")
        print(self.total)

    def add(self, n : int):
        #threading.Thread(target=winsound.Beep, args=(440, 1000)).start()
        self.total += n
        self.send("+"+str(n))
        
    def sub(self, n : int):
        #threading.Thread(target=winsound.Beep, args=(600, 1000)).start()
        self.total -= n
        self.send("-"+str(n))
        
if __name__ == "__main__":
    jnr_server = ConnServer(SERVER_IP, SNRCONNECT_PORT)
    print("Finished!")
    
    while True: # included here to avoid the race condition of the program quitting before anything is actually sent
        n = int(input())
        if n > 0:
            jnr_server.add(n)
        else:
            jnr_server.sub(abs(n))
