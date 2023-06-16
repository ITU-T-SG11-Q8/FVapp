import threading

class GRMQueue:
    lock = None
    Queues = []

    def __init__(self):
        self.lock = threading.Lock()

    def put(self, bin_data):
        self.lock.acquire()
        self.Queues.append(bin_data)
        self.lock.release()

    def pop(self):
        bin_data = None

        self.lock.acquire()
        if len(self.Queues) > 0:
            bin_data = self.Queues.pop(0)
        self.lock.release()

        return bin_data

    def size(self):
        return len(self.Queues)
