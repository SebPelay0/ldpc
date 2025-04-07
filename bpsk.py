
import numpy
import matplotlib.pyplot as plt
import math
import time
import os
import scipy
import scipy.io
import sys
sys.path.append(os.path.abspath("../pyldpc"))
import pyldpc
# numpy.random.seed(29)
# H,G = pyldpc.make_ldpc(8,2, 4,True,True)
FRAME_ERROR = None

class LDPCEncoder():
    def __init__(self, d_v, d_c, n, seed = 20, readDataMatrix= False):
        self.d_v = d_v # number of times each message bit appears in a parity equation 
        self.d_c = d_c # num bits checked in a parity equation // code rate => 1 -(d_v/d_c)
        self.n = n
        self.seed = seed
        self.PN = None
        if readDataMatrix:
            # H,G = readMatrix("parityMatrix.txt")
            H = numpy.array(readMatrixFile("5GMatrix.mat")["H"], dtype=int)
            # H = H[:1080,:] #HALF RATE  ROW REMOVAl
            G = pyldpc.coding_matrix(H)
            self.H = H
            
            self.G = G
            self.m = H.shape[0]
         
            # numpy.savetxt("H.txt", H, fmt='%d')
            # print(H)
            # print(H.shape)
            # plt.figure(figsize=(10, 4))
            # row_weights = numpy.sum(H, axis=1)
            # column_weights = numpy.sum(H, axis=0)
            # plt.bar(range(len(row_weights)), row_weights)
            # plt.xlabel("Row Index")
            # plt.ylabel("Number of 1s")
            # plt.title("Distribution of 1s in Each Row")
            # plt.show()

            # # Plot column weights
            # plt.figure(figsize=(10, 4))
            # plt.bar(range(len(column_weights)), column_weights)
            # plt.xlabel("Column Index")
            # plt.ylabel("Number of 1s")
            # plt.title("Distribution of 1s in Each Column")
            # plt.show()
            print(G.shape)
            
        else:
            H,G = pyldpc.make_ldpc(n,d_v, d_c,True,True)
            self.H = H
            self.G = G
            self.m = n * (d_v/d_c) #num check nodesQQ
        
        nonZero = numpy.count_nonzero(G)
        
        self.y = None
        self.originalEncoded = None
        self.SNR = None
        self.bitEnergyRatio = None
        self.numIterations = None
        self.messageDecoded = None
        self.spread = None
        self.BER = 0
        

    def encode(self, message, snr):
        if(len(message) != self.G.shape[1]):
            print("Invalid Message Length: G is " + str(self.G.shape[1]) + " message is " + str(len(message)))
            return
        
        self.originalEncoded = numpy.dot(self.G, message) % 2
        self.SNR = snr
        noisy = pyldpc.encode(self.G, message, snr)
        # noisy = -1 * self.addNoiseBPSK(snr, self.originalEncoded)
        
        return noisy
    
    def isValidCodeword(self, decoded_codeword):
        syndrome = numpy.dot(self.H, decoded_codeword.T) % 2
        is_valid = numpy.all(syndrome == 0)
        # print(f"syndome {syndrome}")
        return is_valid

    def bitFlipDecode(self, codeword):
        print(f"       \n H matrix: \n {self.H}")
        bitNodes = []
        #make hard decision on initial bit states
        for val in codeword:
            if(val > 0 ):
                bitNodes.append(0)
            else:
                bitNodes.append(1)
        newMessages = [0] * len(bitNodes)
        numIterations = 0
        
        bitNodeMessages = bitNodes.copy()
        print(f"initial received \n{numpy.array(bitNodes)}")
        while numIterations < 10:
            if self.isValidCodeword(numpy.array(bitNodes)):
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
                print(f"Bit flip decoding done after {numIterations} iterations")
                return errors/len(bitNodes)

                break
            bitNodeMessages = bitNodes.copy()
            messagesReceivedByBits = [0] * len(bitNodeMessages)
            #Pass the bit node messages into check nodes
            for j in range(0, int(self.m)):
                Ej = numpy.where(self.H[j] == 1)[0]  #these are the bit nodes that should receive a message from each check node
    
                message = 0
                checkParity = 0
                for i in Ej:
                    checkParity ^= bitNodeMessages[i]  # Compute total check parity

                for target in Ej:
                    parityWithoutBit = checkParity ^ bitNodeMessages[target]  # Remove target bit's effect
                    if parityWithoutBit == 1:
                        messagesReceivedByBits[target] += 1
                    else:
                        messagesReceivedByBits[target] -=1


            for index in range(len(bitNodes)):
                if messagesReceivedByBits[index] > len(Ej) / 2:
                    bitNodes[index] ^= 1  #Flip if majority of checks failed
                    
            print(f"Original code: {self.originalEncoded}")
            print(f"Flipped:       {numpy.array(bitNodes)}")
            numIterations +=1
         
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
        BER = errors/len(bitNodes)
        return BER

    def minSumDecode(self, codeword):
        # codeword = -1  * codeword
        print(f"Codeword: {self.originalEncoded.flatten()}")
        bitNodes = numpy.array(codeword, dtype=float)  # Use soft channel values instead of hard bits]
        
        initialLLRs = []
        numIterations = 0
        # Estimate A Priori LLR's for each bit. 
        for index in range(len(codeword)):
            sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
            if index < 80:
                initialLLRs.append(1e-9)
            else:
                LLR = (2 * float(codeword[index])) / sigma2
                initialLLRs.append(LLR)
        
        # M=> Bit-to-check messages
        M = {}  
        # Initialize bit nodes with the channel LLRs
        for j in range(int(self.m)):
            for i in numpy.where(self.H[j] == 1)[0]:
                if i not in M:
                    M[i] = {}
                M[i][j] = initialLLRs[i]
        
        # for i in range(self.n):
        #     for j in numpy.where(self.H[:, i] == 1)[0]:
        #         M[i][j] = initialLLRs[i]

        #E => Check to bit messages
        E = {j: {} for j in range(int(self.m))}

     

        # # Initialize check-to-bit messages
        
        initialLLRs = numpy.array(initialLLRs, dtype=float)
        bitNodes = initialLLRs.copy()
        hardDecisions = [0]*len(bitNodes)
        for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
    
        BER = 0
        errors = 0
        while numIterations < 30:
            self.numIterations = numIterations
            errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
            print(f"Decoding Iteration {numIterations}: BER {errors/len(bitNodes)}")
            #Test the hard decision on current soft values
            if self.isValidCodeword(numpy.array(hardDecisions)):
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
                print(f"Min Sum Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                self.messageDecoded = hardDecisions
                return errors/len(bitNodes)

            messagesReceivedByBits = numpy.zeros(len(bitNodes)) 

            for j in range(int(self.m)):  
                Ej = numpy.where(self.H[j] == 1)[0]  # All bits connected to check j

                if len(Ej) < 2:
                    continue  # Skip underconnected checks

                for target in Ej:
                    # Use all other bits except the target
                    others = [k for k in Ej if k != target]

                    # Get messages Q[k][j] from other bits to this check
                    incoming = [M[k][j] for k in others]
                    magnitudes = numpy.abs(incoming)
                    signs = numpy.sign(incoming)
                    signs[signs == 0.0] = 1  # Avoid 0 sign

                    minLLR = numpy.min(magnitudes)
                    signProd = numpy.prod(signs)

                    
                    E[j][target] = 0.75 * minLLR * signProd
        
            for i in range(len(bitNodes)):
                incoming_checks = numpy.where(self.H[:, i] == 1)[0]
                bitNodes[i] = initialLLRs[i] + sum(E[j][i] for j in incoming_checks)
                # bitNodes[i] = numpy.clip(bitNodes[i], -1000.0, 1000.0)
                
            for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
            for i in range(self.n):
                for j in numpy.where(self.H[:, i] == 1)[0]:
                    other_checks = [k for k in numpy.where(self.H[:, i] == 1)[0] if k != j]
                    M[i][j] = initialLLRs[i] + sum(E[k][i] for k in other_checks)
            numIterations += 1
        
        
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
        BER = errors/len(bitNodes) 
        print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        self.messageDecoded = bitNodes
        self.BER = BER
        
        errors_mask = (numpy.array(self.originalEncoded) != numpy.array(hardDecisions)).astype(int)
        error_indices = numpy.where(errors_mask == 1)[0]

        

        return FRAME_ERROR
    



    def sumProductDecode(self, codeword):
        # codeword = -1  * codeword
        print(f"Codeword: {self.originalEncoded.flatten()}")
        bitNodes = numpy.array(codeword, dtype=float)  # Use soft channel valus instead of hard bits]
        
        initialLLRs = []
        numIterations = 0
        # Estimate A Priori LLR's for each bit. 
        for index in range(len(codeword)):
            sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
            if index < 80:
                initialLLRs.append(1e-9)
            else:
                LLR = (2 * float(codeword[index])) / sigma2
                initialLLRs.append(LLR)
        
        # M=> Bit-to-check messages
        M = {}  
        # Initialize bit nodes with the channel LLRs
        for j in range(int(self.m)):
            for i in numpy.where(self.H[j] == 1)[0]:
                if i not in M:
                    M[i] = {}
                M[i][j] = initialLLRs[i]
        
        # for i in range(self.n):
        #     for j in numpy.where(self.H[:, i] == 1)[0]:
        #         M[i][j] = initialLLRs[i]

        #E => Check to bit messages
        E = {j: {} for j in range(int(self.m))}

     

        # # Initialize check-to-bit messages
        
        initialLLRs = numpy.array(initialLLRs, dtype=float)
        bitNodes = initialLLRs.copy()
        hardDecisions = [0]*len(bitNodes)
        for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
    
        BER = 0
        errors = 0
        while numIterations < 40:
            self.numIterations = numIterations
            errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
            print(f"Decoding Iteration {numIterations}: BER {errors/len(bitNodes)}")
            #Test the hard decision on current soft values
            if self.isValidCodeword(numpy.array(hardDecisions)):
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
                print(f"Sum Product Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                self.messageDecoded = hardDecisions
                return errors/len(bitNodes)

            messagesReceivedByBits = numpy.zeros(len(bitNodes)) 

            for j in range(int(self.m)):  
                Ej = numpy.where(self.H[j] == 1)[0]  # All bits connected to check j

                for target in Ej:
                    # Use all other bits except the target
                    others = [k for k in Ej if k != target]

                    # Get valid messages bits to this check
                    incoming = [M[k][j] for k in others]
                    signs = []
                    magnitudes = []
                    for val in incoming:
                        magnitudes.append(numpy.abs(val))
                        if val == 0:
                            signs.append(1)
                        else:
                            signs.append(numpy.sign(val))
                    
                    tanhValues = numpy.array([numpy.tanh(M/2) for M in magnitudes])
                    magnitudes = numpy.array(magnitudes)
                    signs = numpy.array(signs)

                    #Calculate product series
                    # product =numpy.prod(signs) * numpy.prod(magnitudes)
                    
                    #message formula
                    phiSum = numpy.sum(numpy.array(self.phi(magnitudes))) #apply all phi(x) on all magnitudes and take sum
                    phiMagnitude = self.phi(phiSum)
                    E[j][target] = numpy.prod(signs) * phiMagnitude
        
                # bitNodes[i] = numpy.clip(bitNodes[i], -100.0, 100.0)
            
            #Set new variable node messages, excluding each check node's own contribution
            for i in range(self.n):
                for j in numpy.where(self.H[:, i] == 1)[0]:
                    other_checks = [k for k in numpy.where(self.H[:, i] == 1)[0] if k != j]
                    M[i][j] = initialLLRs[i] + sum(E[k][i] for k in other_checks)
            
            #Calculate LLR total for the variable node
            for i in range(len(bitNodes)):
                incoming_checks = numpy.where(self.H[:, i] == 1)[0]
                bitNodes[i] = initialLLRs[i] + sum(E[j][i] for j in incoming_checks)

            #Hard decision on each variable node. 
            for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
     
            numIterations += 1
        
        
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
        BER = errors/len(bitNodes) 
        print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        self.messageDecoded = bitNodes
        self.BER = BER
        
        errors_mask = (numpy.array(self.originalEncoded) != numpy.array(hardDecisions)).astype(int)
        error_indices = numpy.where(errors_mask == 1)[0]

        

        return FRAME_ERROR
    

    def phi(self, x):
        #Log approximation of tanh(x)
        # if x == 0:
        #     x = 1e-7
        x = numpy.clip(x, 1e-12,100)
        return -numpy.log(numpy.tanh(x/2))
        # return numpy.log((numpy.exp(x)+ 1)/(numpy.exp(x) -1))

    def addNoiseBPSK(self, SNR_DB, encoded, plot=False):
        power = sum([a**2 for a in encoded]) / len(encoded) 
        
        SNRLinear = 10**(SNR_DB/10)
        noiseStd = numpy.sqrt(1 / ( SNRLinear*2)  ) 
        noise = noiseStd * numpy.random.randn(*encoded.shape)
        
        bpsk = 2 * numpy.array(encoded) - 1  # Convert to -1 and 1

    
        y = bpsk + noise
        self.SNR = SNR_DB
        self.y = y
        self.bitEnergyRatio = power
        self.symbolNoiseRatio = power
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(y)), y, 'r.-', label="Noisy Signal (Received Y)")
            plt.plot(range(len(y)), self.originalEncoded, 'bo-', label="Original BPSK Encoded Signal")
            plt.ylabel("Amplitude")
            plt.xlabel("Bit N")
            plt.title(f"BPSK Signal with AWGN (SNR = {SNR_DB} dB)")
            plt.legend()
            plt.grid(True)
            plt.show()
        return y


    def addNoiseQPSK(self, complexSignal, SNR_DB):
        SNRLinear = 10**(SNR_DB/10)
        power = sum([a**2 for a in complexSignal]) / len(complexSignal) #E_b, for BPSK #E_b == E_s
        
   
        noise = numpy.sqrt(power / (2*SNRLinear)) * (numpy.random.randn(len(complexSignal)) + 1j *(numpy.random.randn(len(complexSignal))))
        y = complexSignal + noise
        print(y)
        plt.figure(figsize=(7, 7))
        plt.scatter(y.real, y.imag, color='blue', marker='x', label="Noisy Symbols")
        plt.scatter(complexSignal.real, complexSignal.imag, color='red', marker='o', label="Original QPSK Symbols", s = 35)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel("Imaginary")
        plt.ylabel("Real")
        plt.title(f"QPSK Constellation with AWGN (SNR = {SNR_DB} dB)")
        plt.legend()
        plt.show()


    def raw(self, codeword):

        hardDecisions = []
        for val in codeword:
            if(val < 0 ):
                hardDecisions.append(0)
            else:
                hardDecisions.append(1)
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
        return errors/len(codeword)
    
    def generateQPSK(self, length):

        QPSKSymbols = numpy.array([1+1j, 1-1j, -1+1j, -1-1j])
        randomSignal = numpy.random.choice(QPSKSymbols, size=length)
        return randomSignal
    
    def write(self, filePath, SNR, BER, sumProd, bitFlip):
        with open(filePath, "a") as file:
            file.write(f"{time.ctime()}: SNR = {SNR}, MinSum={BER}, SumProd{sumProd}, BitFlip{bitFlip}\n")

    def spreadDSS(self, spreadFactor, snr):
        self.PN = numpy.random.randint(0,2, size=spreadFactor)
        finalSpread = []
        # print(f"Pseudo-Random Noise Code: {self.PN}")
        for bit in self.originalEncoded:
            result = [A^bit for A in self.PN]
            finalSpread.append(result)
        # print(f"Final {numpy.array(finalSpread)}")
        self.spread = numpy.array(finalSpread)
        finalSpread = numpy.array(finalSpread)
        noisy = self.addNoiseBPSK(self.SNR, finalSpread, False)
        return noisy
    
    def deSpreadDSS(self, noisy):
        despreadSoftBits = []

        for spreadBitSequence in noisy:
            xorResult = [bit * (1 if pnBit == 1 else -1) for bit, pnBit in zip(spreadBitSequence, self.PN)]
            softValue = sum(xorResult)  
            
            despreadSoftBits.append(softValue)

        return numpy.array(despreadSoftBits)  



        


def readMatrix(filePath):
    with open(filePath, "r") as file:
        lines = file.readlines()
    rows = [list(map(int, line.strip())) for line in lines]

    # Convert to NumPy array
    H = numpy.array(rows, dtype=int)
    print(H.shape)
    # Generate generator matrix G
    G = pyldpc.coding_matrix(H, False)

    # Save G matrix
    with open("gMatrix.txt", "a") as file:
        file.write(f"G: {G}\n")

    return H, G  # Return both matrices
def readMatrixFile(filePath):
    matrix = scipy.io.loadmat(filePath)
    return matrix


#  Expected Noise Std: 0.7071067811865476, Measured Noise Std: 1.0

Test = LDPCEncoder(4,5,2000, readDataMatrix=True)
def test(snr):
    # DSSS Result
    # numpy.random.seed(12)
    print("Starting test...")
    message = numpy.random.randint(0, 2, size=400)
    nonSpread = Test.encode(message, snr)
    noisy = Test.spreadDSS(4, 0)
    codeword = Test.deSpreadDSS(noisy)
    # sumProdEncode = pyldpc.encode(Test.G, message, snr
    print(F"Min Sum Result {Test.minSumDecode(nonSpread)}")
    print(F"Sum Product Result {Test.sumProductDecode(nonSpread)}")
    # print(F"Spread Result {pyldpc.decode(Test.H, sumProdEncode, snr, 30)}")

    #Non-DSSS
    # print(F"Non-Spread Result {Test.minSumDecode(nonSpread)}")
# test(-3.5)
# numpy.random.seed(21)
"""RATE 1/2"""
test0 = LDPCEncoder(2,4, 64)
message0 = numpy.random.randint(0, 2, size=541).tolist()  
test(-3)

""""RATE 3/4"""

# test1 = LDPCEncoder(4,8, 648, readDataMatrix=True)
test1 = LDPCEncoder(4,5,2000, readDataMatrix=True)
# test1 = LDPCEncoder(2,4, 64, readDataMatrix=False)
message1 = numpy.random.randint(0, 2, size=329).tolist()  


""""RATE 1/2"""
test2 = LDPCEncoder(2, 4, 512)
message2 = numpy.random.randint(0, 2, size=519).tolist()  


""""RATE 1/4"""
test3 = LDPCEncoder(3,4, 648)
message3 = numpy.random.randint(0, 2, size=5).tolist()  


def writeData(filePath, BER, frameError):
    with open(filePath, "a") as file:
        string ="BER Values: ["
        string += ", ".join(str(e) for e in BER) 
        string += "]\n"

        frames = "Frame errors: ["
        frames += ", ".join(str(e) for e in frameError) 
        frames += "]\n"
    

def plot(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    snrRange = numpy.arange(-4, 4, 1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    n = 4
    for snr in snrRange:
        avgBER = 0
        avgSumProdBER = 0
        avgBitFlipBER = 0
        frameErrors = 0
        for it in range(n):
            sumProdBER = 0
            BER = 0
            bitFlipBER = 0

            if readMatrixFile:
                test1.H, test1.G = readMatrix("parityMatrix.txt")
            if minSum:
                message1 = numpy.random.randint(0, 2, size=327).tolist()  
                test1.encode(message1) 
                noisyCodeword = test1.addNoiseBPSK(snr)
                # BER = test0.bitFlipDecode(noisyCodeword)
                BER = test1.minSumDecode(noisyCodeword)
            
            if sumProd:
                sumProdEncode = pyldpc.encode(test1.G, message1, snr)
                sumProdDecode = pyldpc.decode(test1.H, sumProdEncode, snr, 30) #maybe change this to use var snr
                sumProdMessage = pyldpc.get_message(test1.G, sumProdDecode)
                sumProdErrors = numpy.sum(numpy.array(message1) != numpy.array(sumProdMessage))
                sumProdBER = sumProdErrors/len(sumProdMessage)  

            if bitFlip:
                message1 = numpy.random.randint(0, 2, size=325).tolist()  
                test1.encode(message1) 
                noisyCodeword = test1.addNoiseBPSK(snr)
                bitFlipBER = 1 - test1.bitFlipDecode(noisyCodeword)
            
            if BER > 0:
                frameErrors +=1
            avgBER += BER
            avgSumProdBER += sumProdBER
            avgBitFlipBER += bitFlipBER
            # if BER == 0:
            #     break
            
        BEROut.append(avgBER/n)
        sumProdBEROut.append(avgSumProdBER/n)
        bitFlipBEROut.append(avgBitFlipBER/n)
        totalFrameErrors.append(frameErrors/n)
        test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
       

    writeData("data.txt", BEROut, totalFrameErrors)
   
    plt.figure(figsize=(8, 5))
    plt.yscale("log")  # Set the Y-axis to logarithmic scale

    plt.plot(snrRange, BEROut, 'b-', label="Min-Sum Decoding")
    plt.plot(snrRange, sumProdBEROut, 'r-', linestyle='-', label=" Sum-Prod Decoding")
    plt.plot(snrRange, bitFlipBEROut, 'g-', linestyle='-', label=" Bit-Flip Decoding")
   
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Decoding: BER vs. SNR at 1/2 Data Rate (n=648, BPSK)")

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frame Error Rate")
    plt.title("LDPC Bit-Flip Decoding: Frame Error vs. SNR at 1/4 Data Rate, n= 256, BPSK")
    plt.grid(True, which="both", linestyle="--")
    plt.show()

    

def plotRates():
    snrRange = numpy.arange(-4, 4, 1)
    BEROut = []
    
    highBEROut = []
    threeQuarterBEROut = []
    halfBEROut = []
    quarterBEROut = []
  
    n = 10
    for snr in snrRange:
        highAvg = 0
        threeQuarterAvg = 0
        halfAverage = 0
        quarterAverage = 0
        for it in range(n):
            ##High Rate
            message0 = numpy.random.randint(0, 2, size=541).tolist()  
            test0.encode(message0)
            highBER = test0.minSumDecode(test0.addNoiseBPSK(snr))
            highAvg += highBER

            # ##Three Quarter
            message1 = numpy.random.randint(0, 2, size=487).tolist()  
            test1.encode(message1)
            threeQuarterBer = test1.minSumDecode(test1.addNoiseBPSK(snr))
            threeQuarterAvg += threeQuarterBer

            ##Half 
            message2 = numpy.random.randint(0, 2, size=257).tolist()  
            test2.encode(message2)
            halfBER = test2.minSumDecode(test2.addNoiseBPSK(snr))
            halfAverage += halfBER

            ##Quarter
            message3 = numpy.random.randint(0, 2, size=164).tolist()  
            test3.encode(message3)
            quarterBER = test3.minSumDecode(test3.addNoiseBPSK(snr))
            quarterAverage += quarterBER

        highBEROut.append(highAvg/n)
        threeQuarterBEROut.append(threeQuarterAvg/n)
        halfBEROut.append(halfAverage/n)
        quarterBEROut.append(quarterAverage/n)



    plt.figure(figsize=(8, 5))
    plt.yscale("log") 

    plt.plot(snrRange, halfBEROut, 'r-', label="R = 1/2")


    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Decoding: BER vs. SNR at 1/2 Data Rate (n=512, BPSK)")

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()

def plotFrameError(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    print("Begin frame error plot")
    snrRange = numpy.array([ -3, -3.0, -2.5,-2.0,-1])

    # snrRange = numpy.arange(-8, -3, 0.1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    maxErrors = 750
    
    for snr in snrRange:
        avgBER = []
        avgSumProdBER = 0
        avgBitFlipBER = 0
        frameErrors = 0
        iterations = 0
        BERS = [0]
        while frameErrors < maxErrors:
            iterations += 1
            os.system("cls")
            print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations}")
            message1 = numpy.random.randint(0, 2, size=400).tolist()  
            
            noist = test1.encode(message1, snr)
            # noisy = test1.spreadDSS(4, snr)
            # codeword = test1.deSpreadDSS(noisy)
            BER =  test1.sumProductDecode(noist)
            if BER != FRAME_ERROR:
                BERS.append(BER)

            # BER = test1.minSumDecode(pyldpc.encode(test1.G, message1, snr))
            if BER is  FRAME_ERROR:
                BERS.append(1)
                frameErrors += 1
            if iterations > 1000:
                # frameErrors = 0
                break
            if iterations == 350 and frameErrors == 0:
                break
        
        totalFrameErrors.append(frameErrors/iterations)


        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    print(f"Total frame errors: {totalFrameErrors}")
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    
    plt.ylabel("Frame Error Rate")
    plt.title("5G LDPC Frame Error vs. SNR at 1/3 Data Rate, n= 2000, z = 80")
    plt.grid(True, which="both", linestyle="--")
    
    plt.show() 
plotFrameError()
