import bpsk
from bpsk import FRAME_ERROR
from loadData import dists
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import sys
from matplotlib.collections import LineCollection
np.random.seed(10) #600 and 10
GOOD = 1
BURST = -1
DISTRIBUTIONS = dists

virtualChannel = bpsk.LDPCEncoder(4,5,2000, readDataMatrix=True, matrixPath="Matrices/5GHalfRate.mat")

message1 = np.random.randint(0, 2, size=400).tolist()  
virtualChannel.encode(message1, 100)
encoded = virtualChannel.originalEncoded



def virtualTransmission(message, distributions, guardBits):
    transmittedValues = []
    i = 0
    scale = 0.85
    guardBits = int(0.20 * len(message))
    while i < len(message) - 1:
        num = random.randint(1,6)
        if message[i] ==0 and message[i+1] ==0:
            if i < guardBits or num> 4:
                complexSymbol = (-1, -1)
            else:
                real = distributions["00"][0].rvs(1)[0]
                imag = distributions["00"][0].rvs(1)[0]
                complexSymbol = (real, imag)
        elif message[i] ==0 and message[i+1] ==1:
            if i < guardBits or num> 4:
                complexSymbol = (-1, 1)
            else:
                real = distributions["01"][0].rvs(1)[0]
                imag = distributions["01"][0].rvs(1)[0]
                complexSymbol = (real, imag)
        elif message[i] ==1 and message[i+1] ==0:
            if i < guardBits or num> 4:
                complexSymbol = (1, -1)
            else:
                real = distributions["10"][0].rvs(1)[0]
                imag = distributions["10"][0].rvs(1)[0]
                complexSymbol = (real, imag)
        elif message[i] ==1 and message[i+1] ==1:
            if i < guardBits or num> 4:
                complexSymbol = (1, 1)
            else:
                real = distributions["11"][0].rvs(1)[0]
                imag = distributions["11"][0].rvs(1)[0]
                complexSymbol = (real, imag)
       
        transmittedValues.append((scale * complexSymbol[0], scale * complexSymbol[1]))

        # print(f"Symbol: {message[i]}{message[i+1]}")
        i+=2
    # print(f"Transmit: {transmittedValues[:10]}")
    # print(f"Message {message}")
    return transmittedValues
def burstTransmission(message, distributions, guardBits, harsh=False):
    transmittedValues = []
    i = 0
    scale = 0.925
    guardBits = int(0.20 * len(message))
    if harsh:
        AVERAGE_STATE_LENGTH = int(len(message)/16)
        BURST_ERROR_PROB = 0.5
        GOOD_ERROR_PROB = 0.2
    else:
        AVERAGE_STATE_LENGTH = int(len(message)/12)
        BURST_ERROR_PROB = 0.25
        GOOD_ERROR_PROB = 0.2

    state = GOOD
    stateLength = int(0.20 * len(message)) #initially assume first 1/5th of frame is free
    while i < len(message) - 1:
        # State transitions
        if stateLength == 0:
            #random roll for new state
            roll = random.randint(1,7)
            if random.random() > 0.4:
                state = GOOD
            else:
                state = BURST
            stateLength = random.randint(1, AVERAGE_STATE_LENGTH)

        symbol = f"{message[i]}{message[i+1]}"

        if state == GOOD:
            if random.random() < GOOD_ERROR_PROB:
                # Occasionally simulate an error even in the good state
                #in this state errors are indpendent samples => less likely for burst
                real = distributions[symbol][0].rvs(1)[0]
                imag = distributions[symbol][1].rvs(1)[0]
                complexSymbol = (real, imag)
            else:
                if symbol == "00":
                    complexSymbol = (-1, -1)
                elif symbol == "01":
                    complexSymbol = (-1, 1)
                elif symbol == "10":
                    complexSymbol = (1, -1)
                elif symbol == "11":
                    complexSymbol = (1, 1)
        if state == BURST:
            if random.random() < BURST_ERROR_PROB:
                # Intentionally corrupt the symbol
                wrong_symbols = ['00', '01', '10', '11']
                wrong_symbols.remove(symbol)
                corrupted_symbol = random.choice(wrong_symbols)
                real = distributions[corrupted_symbol][0].rvs(1)[0]
                imag = distributions[corrupted_symbol][1].rvs(1)[0]
            else:
                if random.random() < 0.25:
                    # Occasionally simulate an error even in the burst state
                    real = distributions[symbol][0].rvs(1)[0]
                    imag = distributions[symbol][1].rvs(1)[0]
                    complexSymbol = (real, imag)
                else:
                    if symbol == "00":
                        complexSymbol = (-1, -1)
                        real = complexSymbol[0]
                        imag = complexSymbol[1]
                    elif symbol == "01":
                        complexSymbol = (-1, 1)
                        real = complexSymbol[0]
                        imag = complexSymbol[1]
                    elif symbol == "10":
                        complexSymbol = (1, -1)
                        real = complexSymbol[0]
                        imag = complexSymbol[1]
                    elif symbol == "11":
                        complexSymbol = (1, 1)
                        real = complexSymbol[0]
                        imag = complexSymbol[1]
                # real = distributions[symbol][0].rvs(1)[0]
                # imag = distributions[symbol][1].rvs(1)[0]
            complexSymbol = (real, imag)

        transmittedValues.append((scale * complexSymbol[0], scale * complexSymbol[1]))
        stateLength -= 1
        i += 2

    # print(f"Transmit (first 10): {transmittedValues[:10]}")
    return transmittedValues


def mapConstellationsToBits(transmittedValues):
    # "00": []  -1 - 1j
    #     "01": -1 + 1j
    #     "10":  1 - 1j
    #     "11":  1 + 1j
    hardDecisions = []
    for real,imag in transmittedValues:
        # print(f"Real {real}, Imag {imag}")
        if real < 0 and imag < 0:
            hardDecisions.append(0)
            hardDecisions.append(0)
        elif real < 0 and imag > 0:
            hardDecisions.append(0)
            hardDecisions.append(1)
        elif real >0 and imag < 0:
            hardDecisions.append(1)
            hardDecisions.append(0)
        elif real > 0 and imag > 0:
            hardDecisions.append(1)
            hardDecisions.append(1)
    return hardDecisions
# print(f"Message {message1}")

def compareOriginalToHardDecisions(original, hardDecisions, received):
    # assert len(original) == len(hardDecisions), "Bitstreams must be equal length."
    error_mags = []
    errors = [1 if original[i] != hardDecisions[i] else 0 for i in range(len(original))]
    if received is not None:
        for i, e in enumerate(errors):
            if e == 1:
                symbol_index = i // 2
                if symbol_index < len(received):
                    magnitude = abs(complex(*received[symbol_index]))
                    error_mags.append(magnitude)
        avg_error_magnitude = np.mean(error_mags) if error_mags else 0
    else:
        avg_error_magnitude = None
    plt.close('all')

    numErrors = sum(errors)

    BER = numErrors/len(original)
    burstCount = 0
    for i in range(1, len(errors) - 1):
        if errors[i] == 1 and (errors[i - 1] == 1 or errors[i + 1] == 1):
            burstCount += 1

    max_burst_length = 0
    current_burst = 0
    for e in errors:
        if e == 1:
            current_burst += 1
            max_burst_length = max(max_burst_length, current_burst)
        else:
            current_burst = 0


    plt.figure(figsize=(12, 3))
    plt.bar(range(len(errors)), errors, color='red', edgecolor='black')
    plt.xlabel("Bit Index")
    plt.title("Simulated Transmission, Channel Errors")
    plt.ylim(-0.1, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    if avg_error_magnitude is not None:
        info = [
            f"BER: {BER:.2f}",
            f"Total Errors: {numErrors}",
            f"Burst Errors: {burstCount}",
            f"Longest Burst: {max_burst_length}",
            f"Avg Error Mag: {avg_error_magnitude:.2f}"
        ]
        plt.plot([], [], ' ', label='\n'.join(info))  # invisible handle
        plt.legend(loc='upper right', frameon=True, bbox_to_anchor=(1.01,1.0), borderaxespad=0)

    plt.tight_layout()
    plt.show()


# transmission = burstTransmission(encoded, DISTRIBUTIONS, 200)
# hardDecisions = mapConstellationsToBits(transmission)
# compareOriginalToHardDecisions(encoded, hardDecisions,transmission)
# virtualChannel.virtualSumProduct(transmission, hardDecisions)
# print(f"Hard {len(hardDecisions)} Original: {len(message1)}")
INTERLEAVE_DEPTH = 13
def interleave(data_bits, depth=INTERLEAVE_DEPTH):
    data_bits = np.asarray(data_bits)
    rows = int(np.ceil(len(data_bits) / depth))
    padded_len = rows * depth
    padded_bits = np.pad(data_bits, (0, padded_len - len(data_bits)), constant_values=0)
    matrix = padded_bits.reshape((rows, depth))
    return matrix.T.flatten()[:len(data_bits)]

def deinterleave(interleaved_bits, depth=INTERLEAVE_DEPTH):
    interleaved_bits = np.asarray(interleaved_bits)
    cols = int(np.ceil(len(interleaved_bits) / depth))
    padded_len = cols * depth
    padded_bits = np.pad(interleaved_bits, (0, padded_len - len(interleaved_bits)), constant_values=0)
    matrix = padded_bits.reshape((depth, cols))
    return matrix.T.flatten()[:len(interleaved_bits)]

def plotBurstError(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    print("Begin frame error plot")
    snrRange = np.array([2.6])

    snrRange = np.arange(1)
    BEROut = []
   
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    maxErrors = 500
    for snr in snrRange:
        avgBER = []
        avgSumProdBER = 0
        avgBitFlipBER = 0
        frameErrors = 0
        iterations = 0
        codeLength = 19968
        accumulateErrors = np.zeros(codeLength)
        totalFrameBitErrors = 0
        failedBitErrors = 0
        BERS = [0]
        while frameErrors < maxErrors:
            iterations += 1
            useInterleave = True
            # input("stop")
            if useInterleave:
                os.system("cls")
                print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations}")
                print(f"SNR RANGE: {snrRange}")
                message1 = np.random.randint(0, 2, size=virtualChannel.G.shape[1]).tolist()  
                
                virtualChannel.encode(message1, 100)
                encoded = virtualChannel.originalEncoded
                print(f"\033[32m Information Length: {virtualChannel.G.shape[1]}, Total Encoded length: {len(encoded)}, Rate: {virtualChannel.G.shape[1]/len(encoded)}\033[0m")
                interleavedEncoded = interleave(encoded)
                transmission = burstTransmission(interleavedEncoded, DISTRIBUTIONS, 200, harsh=True)
                assert (np.array_equal(virtualChannel.originalEncoded, deinterleave(interleavedEncoded))), "INTERLEAVE FAILED"

                print(f"Begin Decoding...")
                hardDecisions = mapConstellationsToBits(transmission)
                # compareOriginalToHardDecisions(virtualChannel.originalEncoded, deinterleave(hardDecisions.copy()),transmission)
                BER = virtualChannel.virtualSumProduct(transmission, deinterleave(hardDecisions), useInterleave)
            else:
                os.system("cls")
                message1 = np.random.randint(0, 2, size=virtualChannel.G.shape[1]).tolist()  
                print(f"Message Length: {virtualChannel.G.shape[1]}")
                virtualChannel.encode(message1, 100)
                encoded = virtualChannel.originalEncoded
                print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations}, RATE: {virtualChannel.G.shape[1]/len(encoded)}")
                print(f"SNR RANGE: {snrRange}")
               
                transmission = burstTransmission(encoded, DISTRIBUTIONS, 200, harsh=True)
                hardDecisions = mapConstellationsToBits(transmission)
                # compareOriginalToHardDecisions(encoded, hardDecisions,transmission)
                BER = virtualChannel.virtualSumProduct(transmission, hardDecisions, useInterleave)

            #Error locations: 
            decoderOutput = virtualChannel.messageDecoded
            assert(len(decoderOutput) == len(encoded), "Decoding length mismatch")
            frameBitErrors =  np.sum(np.array(decoderOutput) != np.array(encoded))
            totalFrameBitErrors += frameBitErrors
            for i, bit in enumerate(decoderOutput):
                if bit != encoded[i]:
                    accumulateErrors[i] +=1
                        
            # noisy = test1.spreadDSS(4, snr)
            # codeword = test1.deSpreadDSS(noisy)
            # BER =  virtualChannel.sumProductDecodeTest(noist)
            if BER != FRAME_ERROR:
                BERS.append(BER)

            # BER = test1.minSumDecode(pyldpc.encode(test1.G, message1, snr))
            if BER is  FRAME_ERROR:
                BERS.append(1)
                frameErrors += 1
                failedBitErrors += frameBitErrors
            if iterations > 50000:
                break
            # if iterations == 200 and frameErrors == 0:
            #     break
            
       
        totalFrameErrors.append(frameErrors/iterations)
        averageErrorsInFailedFrame = failedBitErrors/frameErrors
        averageErrorsInFrame = totalFrameBitErrors/iterations

        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    print(f"SNRS: {snrRange}")
    print(f"Total frame errors: {totalFrameErrors}, Num Iterations: {iterations}")
    filePath = "newErrors.txt"
    with open("C:/Users/lab-user/OneDrive - UNSW/testing/ldpc/Matrices/FER_RESULTS.txt", "a") as f:
        f.write(f"FER: {totalFrameErrors}, Date: {datetime.datetime.now()}, Rate: {virtualChannel.G.shape[1]/len(encoded)}, Encoded Length: {len(encoded)}\n")
    with open(filePath, "a") as file:
            file.write(f"FER: {totalFrameErrors}, Date: {datetime.datetime.now()}\n")
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
   
    plt.ylabel("Frame Error Rate")
    plt.title("Sum Product 5G LDPC Frame Error vs. SNR at 1/5 Data Rate, n= 2000, z = 80")
    plt.grid(True, which="both", linestyle="--")
   
    plt.show()

    nz = np.flatnonzero(accumulateErrors)
    if nz.size == 0:
        print("No errors recorded.")
    else:
        counts = accumulateErrors[nz]
        segs = [((i, 0), (i, c)) for i, c in zip(nz, counts)]

        fig, ax = plt.subplots(figsize=(18, 3))
        lc = LineCollection(segs, linewidths=0.8)
        ax.add_collection(lc)
        ax.scatter(nz, counts, s=6)  # optional dots at the tips
        ax.set_xlim(0, accumulateErrors.size)
        ax.set_ylim(0, counts.max() * 1.05)
        ax.set_xlabel("Bit index")
        ax.set_ylabel("Error count")
        ax.set_title(f"Decoding Error Distribution")
        ax.text(0.99,0.98, f"Average Num Errors: {averageErrorsInFrame}", transform=ax.transAxes, ha="right", va="top", bbox=dict(boxstyle="round", facecolor="white"), zorder=10)
        fig.tight_layout()
        plt.savefig("C:/Users/lab-user/OneDrive - UNSW/testing/ldpc/Matrices/titlePlot.png")
        plt.show() 

plotBurstError()

# plotFrameError()

# def testGuard():
#     frameErrors = 0
#     guardVals = [0,100,200,300,400,500]
#     FER = []
#     for guard in guardVals:
#         iterations = 0
#         frameErrors = 0
#         while frameErrors < 100:
#             message1 = np.random.randint(0, 2, size=400).tolist()  
#             encoded = virtualChannel.encode(message1, 100)
#             transmission = virtualTransmission(encoded, DISTRIBUTIONS, guard)
#             hardDecisions = mapConstellationsToBits(transmission)
#             result = virtualChannel.virtualSumProduct(transmission, hardDecisions)
#             if result == FRAME_ERROR:
#                 frameErrors+=1
#             iterations +=1
#             print(F"FER{frameErrors/iterations}")
#         FER.append(frameErrors/iterations)
# testGuard()