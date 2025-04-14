from bpsk import LDPCEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
FRAME_ERROR = None

def plotFrameError(Test,snrRange, filePath, minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    print("Begin frame error plot")

    # snrRange = numpy.arange(-8, -3, 0.1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    maxErrors = 75
    
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
            print(f"SNR RANGE: {snrRange}")
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
                frameErrors += 1
            if iterations > 100:
                # frameErrors = 0
                break
            if iterations == 200 and frameErrors == 0:
                break
        
        totalFrameErrors.append(frameErrors/iterations)


        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    print(f"SNRS: {snrRange}")
    print(f"Total frame errors: {totalFrameErrors}")
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    with open(filePath, "a") as file:
            file.write(f"SNR RANGE: {snrRange}\nFER: {totalFrameErrors}\n")
    plt.ylabel("Frame Error Rate")
    plt.title("Sum Product 5G LDPC Frame Error vs. SNR at 1/5 Data Rate, n= 2000, z = 80")
    plt.grid(True, which="both", linestyle="--")
    
    plt.show() 


def main():
    print("Enter simulation details...\n")

    while(True):
        try:
            existing = input("Load an existing matrix (.mat) file? Y/N\n");
            if existing == "Y":
                path = input("Enter filepath to matrix\n")
                Test = LDPCEncoder(4,5,2000, readDataMatrix=True)
                selectDecoder = input("For min-sum decoding enter: A \nFor sum-product enter: B\n")
                os.system("cls")
                singleTest = input("Execute a single test? (Y/N)\n")
                if singleTest == "Y":
                    snr = int(float("Select test SNR (DB)\n"))
                    os.system("cls")
                    message = np.random.randint(0, 2, size=400)
                    nonSpread = Test.encode(message, snr)
                    print(F"Sum Product Result {Test.sumProductDecodeTest(nonSpread)}")
                elif singleTest == "N":
                    SNRRange = []
                    filePath = input("Enter file name to store results\n")
                    inputSNR = input("Enter an SNR or exit\n")  
                    while  inputSNR != "exit":
                        SNRRange.append(float(inputSNR))
                        print(f"SNR RANGE {SNRRange}\n")
                        inputSNR = input("Enter an SNR or exit\n")  
                    os.system("cls")
                    print(f"Testing range: {SNRRange}\n")
                    plotFrameError(Test,SNRRange, filePath)

            if input("Exit? (Y/N)") == "Y":
                break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()