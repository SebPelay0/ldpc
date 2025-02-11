import pyldpc
import numpy
import matplotlib.pyplot as plt
import keyboard
import math
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
        self.SNR = 15
        self.bitEnergyRatio = 0
    def encode(self, message, snr=15):
        self.SNR = snr
        self.bitEnergyRatio = snr - 10 * math.log10(1) - 10 * math.log10(0.25) #just random guess for Rb/B
        self.symbolNoiseRatio = self.bitEnergyRatio  * 2
        if(len(message) != self.G.shape[1]):
            print("Invalid Message Length: G is " + str(self.G.shape[1]) + " message is " + str(len(message)))
            return
        
        self.originalEncoded = numpy.dot(self.G, message) % 2

        y = pyldpc.encode(self.G, message, snr, 20)
        #y *= -1
        self.y = y
        return y
    
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
                print(f"Bit flip decoding done after {numIterations} iterations")

                break
            bitNodeMessages = bitNodes.copy()
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
    

    def minSumDecode(self, codeword):
        print(f"\nH matrix: \n{self.H}")
    
        bitNodes = numpy.array(codeword, dtype=float)  # Use LLRs instead of hard bits]
        
        initialLLRs = []
        numIterations = 0
        bitEnergyNoiseRatio = self.SNR/2
        symbolEnergyNoiseRatio = bitEnergyNoiseRatio * 2 # QPSK has 2 bits per symbol

        for index in range(len(codeword)):
            LLR = 2 * codeword[index] * self.symbolNoiseRatio
            initialLLRs.append(LLR)
      
        # Initialize check-to-bit messages
        print(f"initialLLRS {initialLLRs} \n Original {self.originalEncoded}")

        hardDecisions = [0] * len(bitNodes)
        for i in range(len(bitNodes)):
            if initialLLRs[i] > 0:
                hardDecisions[i] = 0
            else: hardDecisions[i] = 1
        print(f"Hard {hardDecisions}")
        BER = 0
        errors = 0
        while numIterations < 30:
        
            if self.isValidCodeword(numpy.array(hardDecisions)):
                print(f"Min Sum Decoding done after {numIterations} Iterations")
                print(f"Iteration {numIterations}: Hard {hardDecisions} \n Original {self.originalEncoded}")
                hardDecode = True
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
                break

            elif self.isValidCodeword(bitNodes):
                print(f"Min Sum Decoding done after {numIterations} Iterations")
                print(f"Soft Bit nodes {bitNodes}")
                for i in range(len(bitNodes)):
                    
                    if bitNodes[i] >  0:
                        bitNodes[i] = 0
                    else: bitNodes[i] = 1
                print(f"Iteration {numIterations}: Bit nodes{bitNodes} \n Original {self.originalEncoded}")
                errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
                break

           
            # if self.isValidCodeword((bitNodes < 0).astype(int)):  # Hard decision on LLRs
            #     print(bitNodes)
            #     print(f"Decoding done {(bitNodes < 0).astype(int)}")
            #     break
            messagesReceivedByBits = [0] * len(bitNodes)
            previousBitNodes = bitNodes.copy()

    
            for j in range(int(self.m)):  
                Ej = numpy.where(self.H[j] == 1)[0]  # Get bit indices connected to check node j
                
                if len(Ej) < 2:
                    continue  # Skip if not enough connections

                # Compute sign product and minimum  LLR excluding target
                messageSign = 1
                for target in Ej:
                    excludeTarget = numpy.setdiff1d(Ej, target)  # Exclude target bit
                    minLLR = numpy.min(bitNodes[excludeTarget]) * 0.75
                    messageSign = numpy.prod(numpy.sign(bitNodes[excludeTarget]))  
                    message = minLLR * messageSign
                    messagesReceivedByBits[target] += message
                    
        
            bitValsTest = []
            for i in range(len(bitNodes)):
                # if messagesReceivedByBits[i] + initialLLRs[i] < 0:
                #     bitNodes[i]  = 1
                # else: bitNodes[i] = 0
                bitNodes[i] =  0.75 * (messagesReceivedByBits[i] + initialLLRs[i])
                
               
                bitValsTest.append(bitNodes[i])
            print(f"Clipped {bitNodes}")
            for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1

            # if numpy.array_equal(previousBitNodes, bitNodes):  # Stop if LLRs converge
            #     print("No changes, stopping early.")
            #     break

            numIterations += 1
        
        if errors != 0:
            errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(hardDecisions))
        BER = errors/len(bitNodes) 
        print(f"Bitnodes{bitNodes}\n")
        print(f"Original{self.originalEncoded}\n")
        print(f"BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        self.write("results.txt", BER,self.bitEnergyRatio )
        return BER
    
    # def minSumDecode(self, codeword):
    #     print(f"\nH matrix: \n{self.H}")
    
    #     bitNodes = numpy.array(codeword, dtype=float)  # Use LLRs instead of hard bits]
        
    #     initialLLRs = []
    #     numIterations = 0
    #     bitEnergyNoiseRatio = self.SNR/2
    #     symbolEnergyNoiseRatio = bitEnergyNoiseRatio * 2 # QPSK has 2 bits per symbol

    #     print(f"Initial received \n{bitNodes}")
    #     for index in range(len(codeword)):
    #         #SNR = 10log(1 / variance) 
    #         LLR = 4 * codeword[index] * symbolEnergyNoiseRatio
    #         initialLLRs.append(LLR)

    #     # Initialize check-to-bit messages
    #     print(f"Initial bitNodes {bitNodes} initialLLRS {initialLLRs}")
    #     hardDecisions = [0] * len(bitNodes)
    #     BER = 0
    #     while numIterations < 30:
    #         if self.isValidCodeword(bitNodes):
    #             print(f"Min Sum Decoding done after {numIterations} Iterations")
                
    #             for i in range(len(bitNodes)):
    #                 if bitNodes[i] > 0:
    #                     bitNodes[i] = 0
    #                 else: bitNodes[i] = 1
    #             print(f"Iteration {numIterations}: Bit nodes{bitNodes} \n Original {self.originalEncoded}")
                
    #             break

    #         if numIterations > 2:
    #             if self.isValidCodeword(numpy.array(hardDecisions)):
    #                 print(f"Min Sum Decoding done after {numIterations} Iterations")
    #                 print(f"Iteration {numIterations}: Hard {hardDecisions} \n Original {self.originalEncoded}")
    #                 print(bitNodes)
                    
    #                 break
    #         # if self.isValidCodeword((bitNodes < 0).astype(int)):  # Hard decision on LLRs
    #         #     print(bitNodes)
    #         #     print(f"Decoding done {(bitNodes < 0).astype(int)}")
    #         #     break
    #         messagesReceivedByBits = [0] * len(bitNodes)
    #         previousBitNodes = bitNodes.copy()

    
    #         for j in range(int(self.m)):  
    #             Ej = numpy.where(self.H[j] == 1)[0]  # Get bit indices connected to check node j
                
    #             if len(Ej) < 2:
    #                 continue  # Skip if not enough connections

    #             # Compute sign product and minimum  LLR excluding target
    #             messageSign = 1
    #             for target in Ej:
    #                 excludeTarget = numpy.setdiff1d(Ej, target)  # Exclude target bit
    #                 minLLR = numpy.min(bitNodes[excludeTarget])
    #                 messageSign = numpy.prod(numpy.sign(bitNodes[excludeTarget]))  
    #                 message = minLLR * messageSign
    #                 messagesReceivedByBits[target] += message
                    
        
    #         bitValsTest = []
    #         for i in range(len(bitNodes)):
    #             # if messagesReceivedByBits[i] + initialLLRs[i] < 0:
    #             #     bitNodes[i]  = 1
    #             # else: bitNodes[i] = 0
    #             bitNodes[i] = messagesReceivedByBits[i] + initialLLRs[i]
               
    #             bitValsTest.append(bitNodes[i])

    #         for i in range(len(bitNodes)):
    #             if bitNodes[i] > 0:
    #                 hardDecisions[i] = 0
    #             else: hardDecisions[i] = 1
    #         # if numpy.array_equal(previousBitNodes, bitNodes):  # Stop if LLRs converge
    #         #     print("No changes, stopping early.")
    #         #     break

    #         numIterations += 1
        
    #     errors = numpy.sum(numpy.array(self.originalEncoded) != numpy.array(bitNodes))
    #     BER = errors/len(bitNodes) 
        
    #     print(f"BER: {BER}")
    #     return BER

    def write(self, filePath, BER, bitEnergyRatio):
        with open(filePath, "a") as file:
            file.write(f"BER = {BER}, SNR={self.SNR}, Eb/N0={bitEnergyRatio}\n")


    

#numpy.random.seed(42)       


"""SHORT MESSAGE HIGH CODE RATE"""
test0 = LDPCEncoder(3,15, 60)
message0 = [1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,0,1]

""""n=200, r = 1/4"""
test1 = LDPCEncoder(15,20, 400)
message1 = numpy.random.randint(0, 2, size=87).tolist()  


""""n=600, r = 1/4"""
test2 = LDPCEncoder(220, 300, 600)
message2 = numpy.random.randint(0, 2, size=519).tolist()  


""""n=600, r = 1/4"""
test3 = LDPCEncoder(225,300, 600)
message3 = numpy.random.randint(0, 2, size=374).tolist()  



# message2 = numpy.random.randint(0, 2, size=519).tolist()  
# noisyCodeword = test3.encode(message3, 9) #snr = 15
# BER = test3.minSumDecode(noisyCodeword)



snrRange = range(15)
BEROut = []
ratios = [ ]
for snr in snrRange:
    avgBER = 0
    for it in range(3):
        message1 = numpy.random.randint(0, 2, size=114).tolist()  
        noisyCodeword = test1.encode(message1, snr) #snr = 15
        BER = test1.minSumDecode(noisyCodeword)
        avgBER += BER
        # if BER == 0:
        #     break
        
    BEROut.append(avgBER/10)
    print(f"Average BER:{avgBER/10}, SNR{snr}")

plt.figure(figsize=(8, 5))
plt.semilogy(snrRange, BEROut, marker='o', linestyle='-')  
plt.xlabel("SNR (dB)")
plt.ylabel("BER (Bit Error Rate)")
plt.title("LDPC Min-Sum Decoding: BER vs. SNR at 1/4 Data Rate, k = 100")
plt.grid(True, which="both", linestyle="--")
plt.show()
