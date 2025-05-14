import bpsk
from bpsk import FRAME_ERROR
from loadData import dists
import random
import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(100) #600 and 10
GOOD = 1
BURST = -1
DISTRIBUTIONS = dists

virtualChannel = bpsk.LDPCEncoder(4,5,2000, readDataMatrix=True)

message1 = np.random.randint(0, 2, size=1000).tolist()  
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
    print(f"Transmit: {transmittedValues[:10]}")
    print(f"Message {message}")
    return transmittedValues

def burstTransmission(message, distributions, guardBits):
    transmittedValues = []
    i = 0
    scale = 0.85
    guardBits = int(0.20 * len(message))
    AVERAGE_STATE_LENGTH = 100
    BURST_ERROR_PROB = 0.25
    GOOD_ERROR_PROB = 0.2

    state = GOOD 
    stateLength = 0.20 * len(message) #initially assume first 1/5th of frame is free
    while i < len(message) - 1:
        # State transitions
        if stateLength == 0:
            #random roll for new state
            roll = random.randint(1,6)
            if roll > 4:
                state = GOOD
            else:
                state = BURST
            stateLength = random.randint(1, AVERAGE_STATE_LENGTH)

        symbol = f"{message[i]}{message[i+1]}"

        if state == GOOD:
            if random.random() < GOOD_ERROR_PROB:
                # Occasionally simulate an error even in the good state
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
                if random.random() > GOOD_ERROR_PROB:
                    # Occasionally simulate an error even in the good state
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
                # real = distributions[symbol][0].rvs(1)[0]
                # imag = distributions[symbol][1].rvs(1)[0]
            complexSymbol = (real, imag)

        transmittedValues.append((scale * complexSymbol[0], scale * complexSymbol[1]))
        stateLength -= 1
        i += 2

    print(f"Transmit (first 10): {transmittedValues[:10]}")
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
    plt.title("Bit Error Map")
    plt.ylim(-0.1, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    if avg_error_magnitude is not None:
        plt.text(
            0.99, 0.85,
            f"Avg Error Magnitude: {avg_error_magnitude:.2f}",
            transform=plt.gca().transAxes,
            fontsize=9,
            ha='right',
            va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
    plt.text(
        0.99, 1.15,
        f"Total Errors: {numErrors}",
        transform=plt.gca().transAxes,
        fontsize=10,
        ha='right',
        va='bottom',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )
    plt.text(
        0.99, 1.05,
        f"Burst Errors: {burstCount}",
        transform=plt.gca().transAxes,
        fontsize=9,
        ha='right',
        va='bottom',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )
    plt.text(
        0.99, 0.95,
        f"Longest Burst Length: {max_burst_length}",
        transform=plt.gca().transAxes,
        fontsize=9,
        ha='right',
        va='bottom',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

    plt.tight_layout()
    plt.show()


transmission = burstTransmission(encoded, DISTRIBUTIONS, 200)
hardDecisions = mapConstellationsToBits(transmission)
compareOriginalToHardDecisions(encoded, hardDecisions,transmission)
virtualChannel.virtualSumProduct(transmission, hardDecisions)
print(f"Hard {len(hardDecisions)} Original: {len(message1)}")



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