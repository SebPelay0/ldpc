from bpsk import LDPCEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
FRAME_ERROR = None


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
WHITE = "\033[37m"
RESET = "\033[0m"
def plotFrameError(Test,snrRange, filePath, maxIterations, maxErrors, minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    print(f"{RED} Begin frame error plot {RESET}")

    # snrRange = numpy.arange(-8, -3, 0.1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    # maxErrors = 75
    
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
            print(f"{WHITE} Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations} {RESET}")
            # print(f"SNR RANGE: {snrRange}")
            message1 = np.random.randint(0, 2, size=Test.G.shape[1]).tolist()  
            
            noist = Test.encode(message1, snr)
            # noisy = test1.spreadDSS(4, snr)
            # codeword = test1.deSpreadDSS(noisy)
            BER =  Test.sumProductDecodeTest(noist)
            if BER != FRAME_ERROR:
                BERS.append(BER)

            # BER = test1.minSumDecode(pyldpc.encode(test1.G, message1, snr))
            if BER is  FRAME_ERROR:
                BERS.append(1)
                print(f"{RED} FAILED {RESET} at transmission no. {iterations} \n")
                frameErrors += 1
            if iterations > maxIterations:
                # frameErrors = 0
                break
            if iterations == 400 and frameErrors == 0:
                break
        
        totalFrameErrors.append(frameErrors/iterations)
        with open(filePath, "w") as file:
            file.write(f"SNR RANGE: {snrRange}\nFER: {totalFrameErrors}\n")

        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    print(f"SNRS: {snrRange}")
    print(f"Total frame errors: {totalFrameErrors}")
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    
    plt.ylabel("Frame Error Rate")
    plt.title("Sum Product 5G LDPC Frame Error vs. SNR at 1/5 Data Rate, n= 2000, z = 80")
    plt.grid(True, which="both", linestyle="--")
    
    plt.show() 


def main():
    print(f"{GREEN} Enter simulation details...\n {RESET}")

    while(True):
        try:
            existing = input(f"Load an existing matrix (.mat) file? Y/N\n ");
            readFileBools = [False,False,False]
            if existing == "Y":
                readFileBools[0] = True
                textFile = input("Enter 'T' to read from a .txt file, 'M' for a .mat file \n")
                readFileBools[1] = (textFile == 'T')

                path = input("\033[37m Enter filepath to matrix\n")
                readFileBools[2] = str(path)
                print(readFileBools)
                Test = LDPCEncoder(4,5,648, readDataMatrix=readFileBools)
                selectDecoder = input(f"{WHITE}For min-sum decoding enter: A \n{WHITE}For sum-product enter: B\n{RESET}")
                os.system("cls")
                singleTest = input("Execute a single test? (Y/N)\n")
                if singleTest == "Y":
                    snr = float(input("Select test SNR (DB)\n"))
                    os.system("cls")
                    message = np.random.randint(0, 2, size=Test.G.shape[1]).tolist() 
                    nonSpread = Test.encode(message, snr)
                    print(F"Sum Product Result {Test.sumProductDecodeTest(nonSpread)}")
                elif singleTest == "N":
                    SNRRange = []
                    filePath = input("Enter file name to store results\n")
                    maxIterations = int(input("Enter max iterations to run\n"))
                    maxErrors = int(input("Enter maximum frame errors per SNR value\n"))
                    inputSNR = input("Enter an SNR or exit\n")
                    while  inputSNR != "exit":
                        SNRRange.append(float(inputSNR))
                        print(f"SNR RANGE {SNRRange}\n")
                        inputSNR = input("Enter an SNR or exit\n")  
                    os.system("cls")
                    print(f"Testing range: {SNRRange}\n")
                    plotFrameError(Test,SNRRange, filePath, maxIterations, maxErrors)

            if input("\033[31m Exit? (Y/N) \033[0m") == "Y":
                break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()