import numpy as np

class MnistDataset:
    def __init__(self, file_name):
        self.infile = open(file_name, 'rb')
        magic_number = int.from_bytes(self.infile.read(4), 'big')
        if(magic_number == 2049):
            self.readLabels()
        elif(magic_number == 2051):
            self.readImages()
        return

    def readLabels(self):
        number_of_items = int.from_bytes(self.infile.read(4), 'big')

        self.datas = np.empty(0)
        for i in range(number_of_items):
            data = int.from_bytes(self.infile.read(1), 'big')
            self.datas = np.append(self.datas, data)
        return

    def readImages(self):
        number_of_images = int.from_bytes(self.infile.read(4), 'big')
        number_of_rows = int.from_bytes(self.infile.read(4), 'big')
        number_of_colmns = int.from_bytes(self.infile.read(4), 'big')
        number_of_pixels = number_of_rows * number_of_colmns

        self.datas = np.empty((0, number_of_pixels))
        for i in range(number_of_images):
            data = np.empty(0)
            for j in range(number_of_pixels):
                data = np.append(data, int.from_bytes(self.infile.read(1), 'big'))
            self.datas = np.append(self.datas, np.array([data]), axis = 0)
        return

    def getDatas(self):
        return self.datas
