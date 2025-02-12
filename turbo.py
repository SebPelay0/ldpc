import numpy
class Turbo:
    def __init__(self, codeword, g1, g2):
        self.g1 = g1
        self.g2 = g2
        self.codeword = codeword
    def encode(self):
        
    def calculateParities(self, generator):
        parities = []
        shiftRegister = [0] * len(generator) #initialise to all zeroes
        checkBits = numpy.where(numpy.array(generator) == 1)[0]
        print(checkBits)
        for val in self.codeword:
            shiftRegister.pop()
            shiftRegister.insert(0, val)
            parityBit = sum([shiftRegister[i] for i in checkBits]) % 2
            parities.append(parityBit)


    def interleave(self)