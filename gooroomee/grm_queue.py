import threading
from collections import deque


class GRMQueue:
    def __init__(self, name='', debug=False):
        #self.lock = None
        #self.lock = threading.Lock()
        self.Queues: deque = deque()
        self.name = name
        self.debug = debug
        # Queues = []

    def clear(self):
        #self.lock.acquire()
        if self.debug is True:
            print(f"clear [{self.name}] queue size:{len(self.Queues)}")
        self.Queues.clear()
        #self.lock.release()

    def put(self, bin_data):
        #self.lock.acquire()
        if self.debug is True:
            print(f"put [{self.name}] queue size:{len(self.Queues)}")
        self.Queues.append(bin_data)
        #self.lock.release()

    def pop(self):
        bin_data = None

        # self.lock.acquire()
        if len(self.Queues) > 0:
            # bin_data = self.Queues.pop(0)
            if self.debug is True:
                print(f"pop [{self.name}] queue size:{len(self.Queues)}")
            bin_data = self.Queues.popleft()
        # self.lock.release()
        return bin_data

    def length(self):
        # self.lock.acquire()
        _length = len(self.Queues)
        if self.debug is True:
            print(f"length [{self.name}] queue size:{len(self.Queues)}")
        # self.lock.release()
        return _length
