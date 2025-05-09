import commpy.modulation
import pyldpc
import numpy
import matplotlib.pyplot as plt
import keyboard
import math
import time
import os

import commpy.modulation as modulate
import commpy


H,G = pyldpc.make_ldpc(8,2, 4,True,True)
FRAME_ERROR = None

class LDPCEncoder():
    def __init__(self, dV, dC, n, seed = 20, readDataMatrix= False):
        self.dV = dV # number of times each message bit appears in a parity equation 
        self.dC = dC # num bits checked in a parity equation // code rate => 1 -(d_v/d_c)
        self.n = n
        self.seed = seed
        self.PN = None
        if readDataMatrix:
            H,G = readMatrix("parityMatrix.txt")
        else:
            H,G = pyldpc.make_ldpc(n,dV, dC,True,True)
        self.H = H
        self.G = G
        self.m = n * (dV/dC) #num check nodes

        self.y = None
        self.originalEncoded = None
        self.SNR = None
        self.bitEnergyRatio = None
        self.numIterations = None
        self.messageDecoded = None
        self.spread = None
        self.BER = 0
        self.originalLength  = 0

    def encode(self, message, snr, M=2):
        if(len(message) != self.G.shape[1]):
            print("Invalid Message Length: G is " + str(self.G.shape[1]) + " message is " + str(len(message)))
            return
        self.originalEncoded = numpy.dot(self.G, message) % 2

        #Standard BPSK
        if M==2:
            self.originalEncoded = numpy.dot(self.G, message) % 2
            self.SNR = snr
            noisy = pyldpc.encode(self.G, message, snr)
            return noisy

        
        #M-QAM
        else:
            self.originalEncoded = numpy.dot(self.G, message) % 2
            self.originalLength = len(self.originalEncoded)
            self.SNR = snr
            modem = modulate.QAMModem(M)
            k = int(numpy.log2(M))
            
            if len(self.originalEncoded) % k != 0:
                print("Encoded message length not divisible by modulation scheme, padding required.")
                padLength = k - (len(self.originalEncoded) % k)  # Compute padding
                self.originalEncoded = numpy.append(self.originalEncoded, numpy.zeros(padLength, dtype=int))

            bitGroups = self.originalEncoded.reshape(-1, k)
            print(f"Original encoded {self.originalEncoded}")
            QAMSymbols = modem.modulate(bitGroups.flatten())
            print(f"Symbols {QAMSymbols}")
            noisy = self.addNoiseQAM(QAMSymbols, snr, plot=False)
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
                # print(f"BER {errors/len(bitNodes)}")
                # print(f"Decoded {bitNodes}")
                print(f"Bit flip decoding done after {numIterations} iterations")
                return errors/len(bitNodes)

                break
            bitNodeMessages = bitNodes.copy()
            messagesReceivedByBits = [0] * len(bitNodeMessages)
            #Pass the bit node messages into check nodes
            for j in range(0, int(self.m)):
                Ej = numpy.where(self.H[j] == 1)[0]  #these are the bit nodes that should receive a message from each check node
                bitNodeIndex = 0
                #calcualte Bi
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

    def minSumDecode(self, codeword, M):
        # codeword = codeword * -1
        initialLLRs =[]
        if M == 2:  # BPSK
            bitNodes = numpy.array(codeword, dtype=float)
            for index in range(len(codeword)):
                sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
                LLR = (2 * codeword[index]) / sigma2

                initialLLRs.append(LLR)
        
        else:  # M-QAM
            initialLLRs = self.calculateLLRS(codeword, M)
        # self.originalEncoded = self.originalEncoded[:self.originalLength]
        initialLLRs = numpy.array(initialLLRs, dtype=float)

        #Bitnodes initialised to a priori LLRS
        bitNodes = initialLLRs.copy()
        numIterations = 0

        #make hard decision on initial llr's 
        hardDecisions = bitNodes.copy()
        BER = 0
        errors = 0
        while numIterations < 10:
            self.numIterations = numIterations

            #Test the hard decision on current soft values
            if self.isValidCodeword(numpy.array(hardDecisions)):
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
                print(f"Min Sum Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                self.messageDecoded = hardDecisions
                print(f"Bitnodes B {bitNodes}")
                return errors/len(bitNodes)

            #Reset message sums to zero 
            messagesReceivedByBits = numpy.zeros(len(bitNodes)) 
    
            for j in range(int(self.m)):  
                Ej = numpy.where(self.H[j] == 1)[0]  # Get bit indices connected to check node j
                
                if len(Ej) < 2:
                    continue  # Skip if not enough connections

                # Compute sign product and minimum  LLR excluding target
                messageSign = 1
                for target in Ej:
                    excludeTarget = numpy.setdiff1d(Ej, target)  # Exclude target bit
                    minLLR = numpy.min(numpy.abs(bitNodes[excludeTarget]))
                    messageSign = numpy.prod(numpy.sign(bitNodes[excludeTarget]))  
                    message = minLLR * messageSign
                    messagesReceivedByBits[target] += message
                    
        
            bitValsTest = []
            for i in range(len(bitNodes)):
                # if messagesReceivedByBits[i] + initialLLRs[i] < 0:
                #     bitNodes[i]  = 1
                # else: bitNodes[i] = 0
                bitNodes[i] =  0.75 * (messagesReceivedByBits[i] + initialLLRs[i])
                
            for i in range(len(bitNodes)):
                if bitNodes[i] < 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1

            # if numpy.array_equal(previousBitNodes, bitNodes):  # Stop if LLRs converge
            #     print("No changes, stopping early.")
            #     break

            numIterations += 1
        
        
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
        BER = errors/len(bitNodes) 
        # print(f"Bitnodes{bitNodes}\n")
        # print(f"Original{self.originalEncoded}\n")
        print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        self.messageDecoded = bitNodes
        self.BER = BER
        print(f"Bitnodes {bitNodes}")
        plt.figure(figsize=(7, 7))
        print(len(bitNodes))
        
        hardBits = numpy.where(bitNodes < 0, 0, 1)
        mistakes = numpy.where(hardBits != self.originalEncoded)[0]  

        plt.plot(numpy.arange(len(mistakes)), mistakes)
        # plt.plot(numpy.arange(len(self.originalEncoded)), self.originalEncoded)
        plt.show()
        return FRAME_ERROR


    def calculateLLRS(self, codeword, M):
        print(f"SNR{self.SNR}")
        modem = modulate.QAMModem(M)  # Create QAM modem
        
        sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
        LLRs = modem.demodulate(codeword, demod_type='soft', noise_var=sigma2)
        
        print(f"len LLRS {len(LLRs)}")
        print(f" LLRS {LLRs}")
        return LLRs
    
    def addNoiseBPSK(self, SNR_DB, encoded, plot=False):
        # print(f"Original encoded message {self.originalEncoded}")
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


    def addNoiseQAM(self,qamSmbols, SNR_dB, plot=False):
        
        SNR_linear = 10**(SNR_dB / 10)
        power = numpy.mean(abs(qamSmbols)**2)  # Proper power calculation
        noise_std = numpy.sqrt(power / (2 * SNR_linear))
        noise = noise_std * (numpy.random.randn(len(qamSmbols)) + 1j * numpy.random.randn(len(qamSmbols)))
        noisy_qam = qamSmbols + noise

        if plot:
            plt.figure(figsize=(7, 7))
            plt.scatter(noisy_qam.real, noisy_qam.imag, color='blue', marker='x', )
            plt.scatter(qamSmbols.real, qamSmbols.imag, color='red', marker='o',  s=35)
            plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.xlabel("Imaginary")
            plt.ylabel("Real")
            plt.title(f"M-QAM Constellation with AWGN (SNR = {SNR_dB} dB)")
            plt.legend()
            plt.show()

        return noisy_qam


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

    def spreadDSS(self, spreadFactor, snr, symbols=[0]):
        if len(symbols) == 1:
            self.PN = numpy.random.randint(0,2, size=spreadFactor)
            finalSpread = []
            # print(f"Pseudo-Random Noise Code: {self.PN}")
            for bit in self.originalEncoded:
                result = [A^bit for A in self.PN]
                finalSpread.append(result)
            # print(f"Final {numpy.array(finalSpread)}")
            self.spread = numpy.array(finalSpread)
            finalSpread = numpy.array(finalSpread)
            print(f"sprad {spreadFactor}")
            noisy = self.addNoiseBPSK(self.SNR, finalSpread, False)
            return noisy
        #QAM Modulation
        else:
            #(±1 values for real & imaginary parts)
            
            self.PN = numpy.random.choice([-1, 1], size=spreadFactor)
            spread_signal = []
            for symbol in symbols:
                spread_symbol = [symbol * pn for pn in self.PN]  # Multiply by PN sequence
                spread_signal.extend(spread_symbol)  
            print(f"Length of spread signal for sf: {spreadFactor}, len={len(spread_signal)}")
            print(f"Spread signal {spread_signal[:3]}")
            return numpy.array(spread_signal)  # Return spread QAM symbols
        
    def deSpreadDSS(self, noisy, spreadFactor, M):
   
        despreadSymbols = []
        if M == 2:
            for spreadBitSequence in noisy:
                xorResult = [bit * (1 if pnBit == 1 else -1) for bit, pnBit in zip(spreadBitSequence, self.PN)]
                softValue = sum(xorResult)  
                
                despreadSymbols.append(softValue)
        else:
            for i in range(0, len(noisy), spreadFactor):
                despread = sum(noisy[i:i+spreadFactor] * self.PN) / spreadFactor  
                despreadSymbols.append(despread)  

        print(f"Despread: {numpy.array(despreadSymbols)}")
        print(len(despreadSymbols))
        return numpy.array(despreadSymbols) 
            


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

# numpy.random.seed(42)
#  Expected Noise Std: 0.7071067811865476, Measured Noise Std: 1.0
numpy.random.seed(42)
Test = LDPCEncoder(4,8,648, readDataMatrix=True)
def test(snr):

    # DSSS Result
    message = numpy.random.randint(0,2,size=324)
    print(message)
    symbols = Test.encode(message, snr, 4)
    # print(f"Original encded {Test.originalEncoded}")
    noisy = Test.spreadDSS(4, snr, symbols)
    # print(f"Result of spread {noisy}")
    demodulated = Test.deSpreadDSS(noisy, 4, 4)
 
    print(F"Spread Result {Test.minSumDecode(demodulated, 4)}")

    #Non-DSSS
    # print(F"Non-Spread Result {Test.minSumDecode(nonSpread)}")

# numpy.random.seed(21)
test(5)
# it = 1
# while test(-2) != FRAME_ERROR:
#     it +=1
# print(it)
"""RATE 1/2"""
test0 = LDPCEncoder(2,4, 64)
message0 = numpy.random.randint(0, 2, size=541).tolist()  


""""BENCHMARK R=1/2"""

# test1 = LDPCEncoder(4,8, 648, readDataMatrix=True)
test1 = LDPCEncoder(4,8,648, readDataMatrix=True)
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
    plt.yscale("log")  # Set the Y-axis to logarithmic scale

    #plt.plot(snrRange, highBEROut, 'b-', label="R = 5/6")
    #plt.plot(snrRange, threeQuarterBEROut, 'g-', label="R = 3/4")
    plt.plot(snrRange, halfBEROut, 'r-', label="R = 1/2")
    #plt.plot(snrRange, quarterBEROut, 'y-', label="R = 1/4")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Decoding: BER vs. SNR at 1/2 Data Rate (n=512, BPSK)")

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()


# def plotFrameError(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
#     snrRange = numpy.arange(-2, 4, 1)
#     BEROut = []
    
#     totalFrameErrors = []
#     sumProdBEROut = []
#     bitFlipBEROut = []
#     maxErrors = 50
#     for snr in snrRange:
#         avgBER = 0
#         avgSumProdBER = 0
#         avgBitFlipBER = 0
#         frameErrors = 0
#         iterations = 0
#         while frameErrors < maxErrors:
#             print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}")
#             iterations += 1
#             message1 = numpy.random.randint(0, 2, size=325).tolist()  
#             test1.encode(message1)
#             BER = test1.minSumDecode(test1.addNoiseBPSK(snr))
#             if BER is FRAME_ERROR:
#                 frameErrors += 1
#             if iterations > 1000:
#                 frameErrors = 0
#                 break
        
#         totalFrameErrors.append(frameErrors/iterations)


#         # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
#     plt.figure(figsize=(8, 5))
#     plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
#     plt.xlabel("SNR (dB)")
#     plt.ylabel("Frame Error Rate")
#     plt.title("LDPC Bit-Flip Decoding: Frame Error vs. SNR at 1/4 Data Rate, n= 256, BPSK")
#     plt.grid(True, which="both", linestyle="--")
#     plt.show()
# plotRates()

# test0.H, test0.G = readMatrix("parityMatrix.txt")

def plotFrameError(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    print("Begin frame error plot")
    snrRange = numpy.array([-8, -7, -6.8, -6.6, -6.4, -6.2, -6, -5, -3])

    # snrRange = numpy.arange(-8, -3, 0.1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    maxErrors = 1000
    
    for snr in snrRange:
        avgBER = []
        avgSumProdBER = 0
        avgBitFlipBER = 0
        frameErrors = 0
        iterations = 0
        BERS = [0]
        while frameErrors < maxErrors:
            iterations += 1
            os.system("clear")
            print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations}")
            
            message1 = numpy.random.randint(0, 2, size=324).tolist()  
            
            test1.encode(message1, snr)
            noisy = test1.spreadDSS(4, snr)
            codeword = test1.deSpreadDSS(noisy)
            BER =  test1.minSumDecode(codeword)
            print(f"Raw result: {test1.raw(codeword)}")
            if BER != FRAME_ERROR:
                BERS.append(BER)

            # BER = test1.minSumDecode(pyldpc.encode(test1.G, message1, snr))
            if BER is  FRAME_ERROR:
                BERS.append(1)
                frameErrors += 1
            if iterations > 12000:
                frameErrors = 0
                break
        
        totalFrameErrors.append(frameErrors/iterations)


        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frame Error Rate")
    plt.title("LDPC short block DSSS: Frame Error vs. SNR at 1/2 Data Rate, n= 6480, BPSK")
    plt.grid(True, which="both", linestyle="--")
    
    plt.show() 
# plotRates()


