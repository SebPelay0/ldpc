import pyldpc
import numpy
import matplotlib.pyplot as plt
import keyboard
import math
import time
H,G = pyldpc.make_ldpc(8,2, 4,True,True)

class LDPCEncoder():
    def __init__(self, d_v, d_c, n, seed = 20):
        self.d_v = d_v # number of times each message bit appears in a parity equation 
        self.d_c = d_c # num bits checked in a parity equation // code rate => 1 -(d_v/d_c)
        self.n = n
        self.seed = seed
        H,G = pyldpc.make_ldpc(n,d_v, d_c,True,True, seed)
        self.H = H
        self.G = G
        self.y = 1
        self.originalEncoded = 1;
        self.m = n * (d_v/d_c)#num check nodes
        self.SNR = 0
        self.bitEnergyRatio = 0
        self.numIterations = 0
        self.messageDecoded = 0
    def encode(self, message):
      
        if(len(message) != self.G.shape[1]):
            print("Invalid Message Length: G is " + str(self.G.shape[1]) + " message is " + str(len(message)))
            return
        
        self.originalEncoded = numpy.dot(self.G, message) % 2
        return self.originalEncoded
    
    def isValidCodeword(self, decoded_codeword):
        syndrome = numpy.dot(self.H, decoded_codeword.T) % 2
        is_valid = numpy.all(syndrome == 0)
        print(f"syndome {syndrome}")
        return is_valid
    

    def bitFlipDecode(self, codeword):
        print(f"       \n H matrix: \n {self.H}")
        bitNodes = []
        bitNodeMessages = []
        #make hard decision on initial bit states
        for val in codeword:
            if(val > 0 ):
                bitNodes.append(0)
            else:
                bitNodes.append(1)
        newMessages = [0] * len(bitNodeMessages)
        numIterations = 0
        print(f"initial received \n{numpy.array(bitNodes)}")
        while numIterations < 30:
            if self.isValidCodeword(numpy.array(bitNodes)):
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
                print(f"BER {errors/len(bitNodes)}")
                print(f"Decoded {bitNodes}")
                print(f"Bit flip decoding done after {numIterations} iterations")
                return errors/len(bitNodes)

                break
            bitNodeMessages = bitNodes[:]
            #Pass the bit node messages into check nodes
            messagesReceivedByBits = [0] * len(bitNodeMessages)
            for j in range(0, int(self.m)):
                Ej = numpy.where(self.H[j] == 1)[0]  #these are the bit nodes that should receive a message from each check
                bitNodeIndex = 0
                #calcualte Bi
                for target in range(len(bitNodeMessages)):
                    message = 0
                    for i in Ej:
                        
                        if target not in Ej: #only check bits attached ot check node
                            continue

                        if i != target: #exclude contribution of target node
                            bitNodeValue = bitNodeMessages[i]
                            message = message ^ bitNodeMessages[i]
                    if(message == 1):
                        messagesReceivedByBits[target] += 1
                    elif target in Ej and message == 0:
                        messagesReceivedByBits[target] -= 1
                    
                    # print(f"Ej{Ej} - Target{target}: Message{message}")
               

            for index in range(len(bitNodes)):
                if messagesReceivedByBits[index] >= 0:
                    bitNodes[index] ^= 1  #Flip if majority of checks failed
                    
            print(f"Original code: {self.originalEncoded}")
            print(f"Flipped:       {numpy.array(bitNodes)}")
            numIterations +=1
         
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
        BER = errors/len(bitNodes)
        return BER

    def minSumDecode(self, codeword):
        print(f"\nH matrix: \n{self.H}")
    
        bitNodes = numpy.array(codeword, dtype=float)  # Use LLRs instead of hard bits]
        
        initialLLRs = []
        numIterations = 0
        for index in range(len(codeword)):
            sigma2 = 1 / (2 * (10**(self.SNR / 10))) 
            LLR = (2 * codeword[index]) / sigma2
            initialLLRs.append(LLR)
      
        # Initialize check-to-bit messages
        print(f"initialLLRS {initialLLRs} \n Original {self.originalEncoded}")

        hardDecisions = [0] * len(bitNodes)
        for i in range(len(bitNodes)):
            if initialLLRs[i] < 0:
                hardDecisions[i] = 0
            else: hardDecisions[i] = 1
        print(f"Hard {hardDecisions}")
        BER = 0
        errors = 0
        while numIterations < 30:
            self.numIterations = numIterations
            if self.isValidCodeword(numpy.array(hardDecisions)):
                print(f"Min Sum Decoding done after {numIterations} Iterations")
                print(f"Iteration {numIterations}: Hard {hardDecisions} \n Original {self.originalEncoded}")
                hardDecode = True
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
                self.messageDecoded = pyldpc.get_message(self.G, hardDecisions)
                print(f"Here {pyldpc.get_message(self.G, hardDecisions)}")
                return errors/len(bitNodes)
       
            elif self.isValidCodeword(bitNodes):
                print(f"Min Sum Decoding done after {numIterations} Iterations")
                print(f"Soft Bit nodes {bitNodes}")
                for i in range(len(bitNodes)):
                    
                    if bitNodes[i] <  0:
                        bitNodes[i] = 0
                    else: bitNodes[i] = 1
                print(f"Iteration {numIterations}: Bit nodes{bitNodes} \n Original {self.originalEncoded}")
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
                self.messageDecoded = pyldpc.get_message(self.G, bitNodes)
                
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
                    messagesReceivedByBits[target] +=( 0.75* message)
                    
        
            bitValsTest = []
            for i in range(len(bitNodes)):
                # if messagesReceivedByBits[i] + initialLLRs[i] < 0:
                #     bitNodes[i]  = 1
                # else: bitNodes[i] = 0
                bitNodes[i] = (messagesReceivedByBits[i] + initialLLRs[i])
                
               
        
            for i in range(len(bitNodes)):
                if messagesReceivedByBits[i] < 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1

            # if numpy.array_equal(previousBitNodes, bitNodes):  # Stop if LLRs converge
            #     print("No changes, stopping early.")
            #     break

            numIterations += 1
        
        
        errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
        BER = errors/len(bitNodes) 
        for i in range(len(bitNodes)):   
            if bitNodes[i] < 0:
                bitNodes[i] = 0
            else: bitNodes[i] = 1
        self.messageDecoded = pyldpc.get_message(self.G, bitNodes)
        print(f"Bitnodes{bitNodes}\n")
        print(f"Original{self.originalEncoded}\n")
        print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        return BER

    def addNoiseBPSK(self, SNR_DB, plot=False):
        # print(f"Original encoded message {self.originalEncoded}")
        power = sum([a**2 for a in self.originalEncoded]) / len(self.originalEncoded) #E_b, for BPSK #E_b == E_s
        
        SNRLinear = 10**(SNR_DB/10)
        noiseStd = numpy.sqrt(1 / (2*SNRLinear))
        noise = noiseStd * numpy.random.randn(len(self.originalEncoded))
        # print(f"Noise={noise}")
        bpsk = 2 * numpy.array(self.originalEncoded) - 1  # Convert to -1 and 1

    
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



    def generateQPSK(self, length):

        QPSKSymbols = numpy.array([1+1j, 1-1j, -1+1j, -1-1j])
        randomSignal = numpy.random.choice(QPSKSymbols, size=length)
        return randomSignal
    
    def write(self, filePath, BER, frame):
        with open(filePath, "a") as file:
            file.write(f"{time.ctime()}: BER = {BER}, SNR={self.SNR}, frameErrors{frame}\n")





def readMatrix(filePath):
    with open(filePath, "r") as file:
        lines = file.readlines()

    rows = []
    for line in lines:
        row = list(map(int, line.strip())) #convert each char to int
        rows.append(row)
        
    H = numpy.array(rows)
    G = pyldpc.coding_matrix(H, True)
    
  
    return H,G

# numpy.random.seed(42)       


"""SHORT MESSAGE 1/3 CODE RATE"""
test0 = LDPCEncoder(2,4, 256)
message0 = numpy.random.randint(0, 2, size=129).tolist()  
""""n=200, r = 1/4"""
test1 = LDPCEncoder(5,6, 648)
message1 = numpy.random.randint(0, 2, size=325).tolist()  


""""n=600, r = 1/4"""
test2 = LDPCEncoder(220, 300, 600)
message2 = numpy.random.randint(0, 2, size=519).tolist()  


""""n=600, r = 1/4"""
test3 = LDPCEncoder(2,4, 8)
message3 = numpy.random.randint(0, 2, size=5).tolist()  

# test0.encode(message0)
# noisyCodeword = test0.addNoiseBPSK(1, True)
# print(test0.minSumDecode(noisyCodeword))

# message2 = numpy.random.randint(0, 2, size=519).tolist()  
# noisyCodeword = test3.encode(message3, 9) #snr = 15
# BER = test3.minSumDecode(noisyCodeword)
print(f"Message original {message0}")
test0.encode(message0) 
noisyCodeword = test0.addNoiseBPSK(2, False)
print(test0.bitFlipDecode(noisyCodeword))

print(f"Decoded {test0.messageDecoded}")
# print("HERE")
# print(test1.minSumDecode(noisyCodeword))
# print("HERE1") #0.11574074074074074
# sumProdEncode = pyldpc.encode(test0.G, message0, -2)
# sumProdDecode = pyldpc.decode(test0.H, sumProdEncode, -2, 30)
# sumProdMessage = pyldpc.get_message(test0.G, sumProdDecode)
# sumProdErrors = numpy.sum(numpy.array(message0) != numpy.array(sumProdMessage))
# sumProdBER = sumProdErrors/len(message0)     
# print(f"Message {message0}")
# print(f"SumProdMessage{sumProdMessage}")
# print(f"BER{sumProdBER} , length{len(message0)}")


def writeData(filePath, BER, frameError):
    
    with open(filePath, "a") as file:
        string ="BER Values: ["
        string += ", ".join(str(e) for e in BER) 
        string += "]\n"

        frames = "Frame errors: ["
        frames += ", ".join(str(e) for e in frameError) 
        frames += "]\n"

def plot():
    snrRange = numpy.arange(-6, 1, 1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    n = 10
    for snr in snrRange:
        avgBER = 0
        avgSumProdBER = 0
        frameErrors = 0
        for it in range(n):
            message1 = numpy.random.randint(0, 2, size=112).tolist()  
            test1.encode(message1) 
            noisyCodeword = test1.addNoiseBPSK(snr)
            BER = test1.minSumDecode(noisyCodeword)
            # sumProdEncode = pyldpc.encode(test1.G, message1, snr)
            # sumProdDecode = pyldpc.decode(test1.H, sumProdEncode, 10, 50) #maybe change this to use var snr
            # sumProdMessage = pyldpc.get_message(test1.G, sumProdDecode)
            # sumProdErrors = numpy.sum(numpy.array(message1) != numpy.array(sumProdMessage))
            # sumProdBER = sumProdErrors/len(sumProdMessage)  
            if BER > 0:
                frameErrors +=1
            avgBER += BER
            avgSumProdBER += 0
            # if BER == 0:
            #     break
            
        BEROut.append(avgBER/n)
        sumProdBEROut.append(avgSumProdBER/n)
        totalFrameErrors.append(frameErrors/n)
        test1.write("results2.txt", avgBER/n, frameErrors/n)
        print(f"Average BER:{avgBER/n}, SNR{snr}")

    writeData("data.txt", BEROut, totalFrameErrors)
   
    plt.figure(figsize=(8, 5))
    plt.yscale("log")  # Set the Y-axis to logarithmic scale

    plt.plot(snrRange, BEROut, 'r.-', label="Min-Sum Decoding")

   
    #plt.plot(snrRange, sumProdBEROut, 'bo-', linestyle='-', label="Sum-Product Decoding")

   
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Decoding: BER vs. SNR at 1/6 Data Rate (n=648, BPSK)")

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frame Error Rate")
    plt.title("LDPC Min-Sum Decoding: Frame Error vs. SNR at 1/6 Data Rate, n= 648, BPSK")
    plt.grid(True, which="both", linestyle="--")
    plt.show()

    


  







# test0.H, test0.G = readMatrix("parityMatrix.txt")
# plot()

