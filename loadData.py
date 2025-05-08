import numpy as np
import os
import matplotlib.pyplot as plt
import sys

interpolate = False
def loadArray(filepath, plot=False):
    data = np.load(filepath)
  
    if plot:
        ideal = np.array([1+1j, -1+1j, -1-1j, 1-1j])

        plt.figure(figsize=(7, 7))
        plt.scatter(data.real, data.imag, color='blue', marker='x', label='Received Symbols')
        plt.scatter(ideal.real, ideal.imag, color='red', marker='o', s=60, label="Ideal QPSK")

        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)  
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.legend()
        plt.ylim(-1.5,1.5)

        plt.show()

    return data

import os

def getFiles(base_dir):
    paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npy"):
                full_rel_path = os.path.join(root, file)
             
                rel_path = os.path.relpath(full_rel_path, base_dir)
                paths.append(rel_path)
    return paths

baseDirPath = "conditionalData/conditional constellation points/single carrier_1744853468"


def sortFiles(baseDir, files, interpolated):
    symbols = {
        "00": [[],[],[],[],[]], # -1 - 1j
        "01": [[],[],[],[],[]], #-1 + 1j
        "10": [[],[],[],[],[]], # 1 - 1j
        "11": [[],[],[],[],[]]# 1 + 1j
    }

    for file in files:
        if (interpolated and file.find("interp")) or ((not interpolated) and file.find("scfde")):
            frameSplit = file.split("frame")[-1]
            frameNum = int(frameSplit.split(".npy")[0])
            if "minus_one" in file and "plus_jone" in file:
                symbols["01"][frameNum].append(loadArray(baseDir+ "/"+ file))
            elif "minus_one" in file and "minus_jone" in file:
                symbols["00"][frameNum].append(loadArray(baseDir+ "/"+ file))

            elif "one_minus_jone_" in file:
                symbols["10"][frameNum].append(loadArray(baseDir+ "/"+ file))
            elif "one_plus_jone" in file:
                symbols["11"][frameNum].append(loadArray(baseDir+ "/"+ file))
    total_00 = len(symbols["00"][4][0])
    print(symbols["00"][0])
    print(f"Count {total_00}")
    # sys.exit()
    return symbols


def getDistribution(frames):
    
    received = {
        "00": [], # -1 - 1j
        "01": [], # -1 + 1j
        "10": [], #  1 - 1j
        "11": []# 1 + 1j
    }
    for frame in frames:
        for point in frame:
            if point.real < 0 and point.imag < 0:
                received["00"].append(point)
            elif point.real < 0 and point.imag > 0:
                received["01"].append(point)
            elif point.real > 0 and point.imag < 0:
                received["10"].append(point)
            elif point.real >0 and point.imag > 0:
                received["11"].append(point)
    return received

files = getFiles(baseDirPath)

symbols = sortFiles(baseDirPath, files, interpolated=interpolate)


received = getDistribution(symbols["00"][0])
def plotFrameDistribution(symbols, baseDirPath, interpolated):
    folderName = baseDirPath.split("/")[-1]
    # print(baseDirPath.split("/")[-1])
    #sys.exit()
    bitPatterns = ["00", "01", "10", "11"]
    frames = range(0,5)
    for bitPattern in bitPatterns:
        for frame in frames:
            
            received = getDistribution(symbols[bitPattern][frame])
            numCorrect = len(received[bitPattern])
            total = sum(len(v) for v in received.values())
            errorRate = 1 - (numCorrect/total)

            #Histogram plot
            labels = list(received.keys())
            counts = [len(received[k]) for k in labels]

            plt.figure(figsize=(6, 4))
            plt.bar(labels, counts, color='skyblue', edgecolor='black')
            plt.title(f"Received Symbol Distribution for Transmitted '{bitPattern}', Frame {frame}")
            plt.xlabel("Detected Symbol")
            plt.ylabel("Transmitted Symbol Count")
            plt.grid(axis='y', linestyle='--', linewidth=0.5)
            plt.ylim(0, max(counts) + 10)
            for i, count in enumerate(counts):
                plt.text(i, count + 2, str(count), ha='center', va='bottom', fontsize=9)
            plt.text(
                2.5,                        
                max(counts) * 0.75   ,            
                f"Error Rate: {errorRate*100:.2f}%",
                ha='center', va='bottom',
                fontsize=10, color='red'
            )

            mode = "interpolated" if interpolated else "scfde"
            output_dir = os.path.join("Distributions", folderName, mode)

            os.makedirs(output_dir, exist_ok=True)

            filename = f"{bitPattern}_frame{frame}.png"
            filepath = os.path.join(output_dir, filename)

            plt.savefig(filepath)
            plt.close()


            
for mapping, points in received.items():
    print(f"Mapping {mapping}: {len(points)} points")


# plotFrameDistribution(symbols, baseDirPath, interpolated=interpolate)

path = "conditional constellation points/single carrier_1744853468/minus_one_minus_jone_rx_scfde_frame0.npy"
# data = loadArray(path)

# data = loadArray("/home/sebastian/LDPC/ldpc/conditionalData/conditional constellation points/single carrier_1744853780/one_minus_jone_rx_scfde_frame1.npy", plot=True)