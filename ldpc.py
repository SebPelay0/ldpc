import pyldpc
import numpy
import matplotlib.pyplot as plt
import keyboard
import math
import time
import os

H,G = pyldpc.make_ldpc(8,2, 4,True,True)
FRAME_ERROR = None

class LDPCEncoder():
    def __init__(self, d_v, d_c, n, seed = 20, readDataMatrix= False):
        self.d_v = d_v # number of times each message bit appears in a parity equation 
        self.d_c = d_c # num bits checked in a parity equation // code rate => 1 -(d_v/d_c)
        self.n = n
        self.seed = seed
        self.PN = None
        if readDataMatrix:
            H,G = readMatrix("parityMatrix.txt")
           
        else:
            H,G = pyldpc.make_ldpc(n,d_v, d_c,True,True, seed)
            
            #print(f"DV: {dv_distribution}, DC: {dc_distribution}")
        self.H = H
        self.G = G
        self.y = 1
        self.originalEncoded = 1;
        self.m = n * (d_v/d_c)#num check nodes
        self.SNR = 0
        self.bitEnergyRatio = 0
        self.numIterations = 0
        self.messageDecoded = 0
        self.spread = None

    def encode(self, message, snr):
      
        if(len(message) != self.G.shape[1]):
            print("Invalid Message Length: G is " + str(self.G.shape[1]) + " message is " + str(len(message)))
            return
        
        self.originalEncoded = numpy.dot(self.G, message) % 2
        self.SNR = snr
        noisy = pyldpc.encode(self.G, message, snr)
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
            #Pass the bit node messages into check nodes
            messagesReceivedByBits = [0] * len(bitNodeMessages)
            for j in range(0, int(self.m)):
                Ej = numpy.where(self.H[j] == 1)[0]  #these are the bit nodes that should receive a message from each check
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

    def minSumDecode(self, codeword):
        # print(f"Codeword {codeword}")
    
        bitNodes = numpy.array(codeword, dtype=float)  # Use LLRs instead of hard bits]
        
        initialLLRs = []
        numIterations = 0
        bitEnergyNoiseRatio = self.SNR/2
        symbolEnergyNoiseRatio = bitEnergyNoiseRatio * 2 # QPSK has 2 bits per symbol

        for index in range(len(codeword)):
            sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
            LLR = (2 * codeword[index]) / sigma2
            initialLLRs.append(LLR)
        # Initialize check-to-bit messages
        # print(f"initialLLRS {initialLLRs} \n Original {self.originalEncoded}")

        hardDecisions = bitNodes.copy()
        # print(f"Bitnodes {bitNodes}")
        # for i in range(len(bitNodes)):
        #     if initialLLRs[i] > 0:
        #         hardDecisions[i] = 0
        #     else: hardDecisions[i] = 1
        # print(f"Hard {hardDecisions}")
        BER = 0
        errors = 0
        while numIterations < 30:
            self.numIterations = numIterations
            if self.isValidCodeword(numpy.array(hardDecisions)):
                
                # print(f"Iteration {numIterations}: Hard {hardDecisions} \n Original {self.originalEncoded}")
                hardDecode = True
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
                # print(f"Original encode {self.originalEncoded}")
                print(f"Min Sum Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                self.messageDecoded = hardDecisions
                return errors/len(bitNodes)

            elif self.isValidCodeword(bitNodes):
              
                # print(f"Soft Bit nodes {bitNodes}")
                for i in range(len(bitNodes)):
                    
                    if bitNodes[i] >  0:
                        bitNodes[i] = 0
                    else: bitNodes[i] = 1
                # print(f"Iteration {numIterations}: Bit nodes{bitNodes} \n Original {self.originalEncoded}")
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
                self.messageDecoded = bitNodes
                print(f"Min Sum Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                return errors/len(bitNodes)
                

           
            # if self.isValidCodeword((bitNodes < 0).astype(int)):  # Hard decision on LLRs
            #     print(bitNodes)
            #     print(f"Decoding done {(bitNodes < 0).astype(int)}")
            #     break
            messagesReceivedByBits = numpy.zeros(len(bitNodes)) 

            previousBitNodes = bitNodes.copy()

    
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
                if bitNodes[i] > 0:
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
        return FRAME_ERROR

    def addNoiseBPSK(self, SNR_DB, encoded, plot=False):
        # print(f"Original encoded message {self.originalEncoded}")
        power = sum([a**2 for a in encoded]) / len(encoded) #E_b, for BPSK #E_b == E_s
        
        SNRLinear = 10**(SNR_DB/10)
        noiseStd = numpy.sqrt(1 / ( SNRLinear/ 0.5)  ) # add 2 * SNR Linear
        noise = noiseStd * numpy.random.randn(*encoded.shape)
        measured_noise_std = numpy.std(self.y - (2 * numpy.array(encoded) - 1))
        # print(f"Expected Noise Std: {noiseStd}, Measured Noise Std: {measured_noise_std}")
        # print(noise[0:10])
        # print(f"Noise={noise}")
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
        
        #Maybe remove 1/2 in sqrt?
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
        # print(f"Original {self.originalEncoded}")
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

# numpy.random.seed(42)
#  Expected Noise Std: 0.7071067811865476, Measured Noise Std: 1.0
Test = LDPCEncoder(2,4,8, readDataMatrix=True)
def test(snr):
    message = numpy.random.randint(0,2,size=324)
    Test.encode(message, snr)
    noisy = Test.spreadDSS(4, 0)
    print(f"Noisy {noisy}")
    codeword = Test.deSpreadDSS(noisy)
    print(f"Despread {codeword}")
    return Test.minSumDecode(codeword)

# [[ 1.4040456  -0.72009848 -0.96278865  1.14014006]
#  [-2.58890927  0.50195388  0.83554252 -0.76993264]
#  [-0.9480784   0.84935142  0.76418154 -1.11269413]
#  [ 2.44561307 -1.21335093 -1.66811847  0.85274585]
#  [-1.04340024  0.58270395  1.24609292 -1.76272888]
#  [ 0.42669303 -1.78342084 -1.5048976   1.09509052]
#  [-0.26237081  0.56125471  0.6782352  -1.20025154]
#  [ 0.22591687 -1.83990624 -1.20982926 -0.14128704]]

# it = 1
# while test(1) != FRAME_ERROR:
#     it +=1
# print(it)
"""RATE 5/6"""
test0 = LDPCEncoder(2,12, 648)
message0 = numpy.random.randint(0, 2, size=541).tolist()  


""""RATE 3/4"""

test1 = LDPCEncoder(4,8, 648, readDataMatrix=True)
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
    snrRange = numpy.arange(-6, -5, 0.5)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    maxErrors = 10
    for snr in snrRange:
        avgBER = 0
        avgSumProdBER = 0
        avgBitFlipBER = 0
        frameErrors = 0
        iterations = 0
        while frameErrors < maxErrors:
            iterations += 1
            os.system("clear")
            print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations}")
            message1 = numpy.random.randint(0, 2, size=324).tolist()  
            
            test1.encode(message1, snr)
            noisy = test1.spreadDSS(4, snr)
            codeword = test1.deSpreadDSS(noisy)
            BER =  test1.minSumDecode(codeword)
           
            

            # BER = test1.minSumDecode(pyldpc.encode(test1.G, message1, snr))
            if BER is FRAME_ERROR:
                frameErrors += 1
            if iterations > 750:
                frameErrors = 0
                break
        
        totalFrameErrors.append(frameErrors/iterations)


        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frame Error Rate")
    plt.title("LDPC Bit-Flip Decoding: Frame Error vs. SNR at 1/2 Data Rate, n= 648, BPSK")
    plt.grid(True, which="both", linestyle="--")
    
    plt.show() 
# plotRates()
# plotFrameError()


# test0.H, test0.G = readMatrix("parityMatrix.txt")



