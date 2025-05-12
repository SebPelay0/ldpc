import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import scipy.stats

interpolate = False
baseDirPath = "conditionalData/conditional constellation points/single carrier_1744853780"

def loadArray(filepath, plot=False):
    data = np.load(filepath)
    # print(data)
    print(f"Length {len(data)}")
    # sys.exit()
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


def getFiles(base_dir):
    paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npy"):
                full_rel_path = os.path.join(root, file)
             
                rel_path = os.path.relpath(full_rel_path, base_dir)
                paths.append(rel_path)
            
    return paths

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
    # print(symbols["00"][0])
    # print(f"Count {total_00}")
    # sys.exit()
    return symbols

def plotMagnitudes(result, bitPattern):
    data = result[bitPattern]
    if not data:
        print(f"No data for pattern {bitPattern}")
        return

    reals = [pt.real for pt in data]
    imags = [pt.imag for pt in data]

    plt.figure(figsize=(10, 4))

    #magnitude of real components
    plt.subplot(1, 2, 1)
    plt.hist(reals, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Real Values for Pattern {bitPattern}")
    plt.xlabel("Real Part")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', linewidth=0.5)

    #imaginary magnitudes
    plt.subplot(1, 2, 2)
    plt.hist(imags, bins=30, color='salmon', edgecolor='black')
    plt.title(f"Imaginary Values for Pattern {bitPattern}")
    plt.xlabel("Imaginary Part")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def getDistribution(frames, single =False):
    
    received = {
        "00": [], # -1 - 1j
        "01": [], # -1 + 1j
        "10": [], #  1 - 1j
        "11": []# 1 + 1j
    }

    if single:
        for point in frames:
            if point.real < 0 and point.imag < 0:
                received["00"].append(point)
            elif point.real < 0 and point.imag > 0:
                received["01"].append(point)
            elif point.real > 0 and point.imag < 0:
                received["10"].append(point)
            elif point.real >0 and point.imag > 0:
                received["11"].append(point)
    else:
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


def plotFrameDistribution(symbols, baseDirPath, interpolated, save=True):
    folderName = baseDirPath.split("/")[-1]
    
    # print(baseDirPath.split("/")[-1])
    # sys.exit()
    bitPatterns = ["00", "01", "10", "11"]
    frames = range(0,5)
    normalDistributions = dict()
    for bitPattern in bitPatterns:
        combinedFrame  = []
        
        for frame in frames:
            combinedFrame.extend(symbols[bitPattern][frame])
            received = getDistribution(symbols[bitPattern][frame])
            numCorrect = len(received[bitPattern])
            total = sum(len(v) for v in received.values())
            errorRate = 1 - (numCorrect/total)
            conditionalProbs = {k: len(received[k]) / total if total > 0 else 0 for k in received}

            labels = list(received.keys())
            counts = [len(received[k]) for k in labels]

            plt.figure(figsize=(6, 4))
            plt.bar(labels, counts, color='skyblue', edgecolor='black')
            plt.title(f"Received Symbol Distribution for Transmitted '{bitPattern}', Frame {frame}")
            for i, count in enumerate(counts):
                prob = count / total
                rx_label = labels[i]
                
                
                plt.text(i, 75, f"P({rx_label} | {bitPattern}) = {prob*100:.1f}%", 
                        ha='center', va='bottom', fontsize=6, color='blue', )
            
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
            if save:
                plt.savefig(filepath)
            plt.close()

            received_points = np.concatenate(symbols[bitPattern][frame])  # 1D complex array

            plt.figure(figsize=(10, 4))

            # Real part histogram
            plt.subplot(1, 2, 1)
            plt.hist(received_points.real, color='skyblue', edgecolor='black', bins=20)
            plt.title(f"Real Component | Transmitted '{bitPattern}', Frame {frame}")
            plt.xlabel("Real Part")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', linewidth=0.5)

            # Imaginary part histogram
            plt.subplot(1, 2, 2)
            plt.hist(received_points.imag, color='lightcoral', edgecolor='black', bins=20)
            plt.title(f"Imaginary Compnent | Transmitted '{bitPattern}', Frame {frame}")
            plt.xlabel("Imaginary Part")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', linewidth=0.5)

            plt.suptitle(f"'{bitPattern}', Frame {frame}")
            # plt.show()
            plt.close()

        #Combine all frames within for a transmitted symbol 
        plt.figure(figsize=(10, 4))

        # Real part histogram + PDF
        received = getDistribution(combinedFrame)
        received_points = np.concatenate(combinedFrame)  # 1D complex array
        realVals = received_points.real
        imagVals = received_points.imag
        realMean, imagMean = np.mean(realVals), np.mean(imagVals)
        stdReal, stdImag = np.std(realVals,ddof=1), np.std(imagVals, ddof=1)

        #since frame data is approximately normal
        realDist = scipy.stats.norm(realMean, stdReal)
        imagDist = scipy.stats.norm(imagMean, stdImag)
        normalDistributions[bitPattern] = [realDist, imagDist]
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(realVals, bins=20, density=True, color='skyblue', edgecolor='black', alpha=0.6)

        x_real = np.linspace(min(realVals), max(realVals), 200)
        plt.plot(x_real, realDist.pdf(x_real), 'k--', label='Normal Fit')

        plt.title(f"Real Component | '{bitPattern}' (All Frames)")
        plt.xlabel("Real Part")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.text(0.05, 0.95,
                f"$\mu$ = {realMean:.2f}\n$\sigma^2$ = {stdReal**2:.2f}",
                transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        # Imaginary part histogram + PDF
        plt.subplot(1, 2, 2)
        plt.hist(imagVals, bins=20, density=True, color='lightcoral', edgecolor='black', alpha=0.6)

        x_imag = np.linspace(min(imagVals), max(imagVals), 200)
        plt.plot(x_imag, imagDist.pdf(x_imag), 'k--', label='Normal Fit')

        plt.title(f"Imaginary Component | '{bitPattern}' (All Frames)")
        plt.xlabel("Imaginary Part")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.text(0.05, 0.95,
                f"$\mu$ = {imagMean:.2f}\n$\sigma^2$ = {stdImag**2:.2f}",
                transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        plt.suptitle(f"Fitted Distribution for '{bitPattern}' (All Frames Combined)")

        # Save and close
        hist_filename = f"{bitPattern}_normal_fit.png"
        hist_filepath = os.path.join(output_dir, hist_filename)
        if save:
            print(f"Saving at {hist_filepath}")
            plt.savefig(hist_filepath)
        # plt.show()
        plt.close()
    
    return normalDistributions

import numpy as np
import matplotlib.pyplot as plt

def printErrorsWithinFrame(frames, symbol):
    errors = []
    combinedFrame = np.concatenate(frames)
    frame_lengths = [len(f) for f in frames]
    frame_ends = np.cumsum(frame_lengths)[:-1]

    # Determine symbol error
    for point in combinedFrame:
        if symbol == "00":
            errors.append(0 if (point.real < 0 and point.imag < 0) else 1)
        elif symbol == "01":
            errors.append(0 if (point.real < 0 and point.imag > 0) else 1)

    max_burst_length = 0
    current_burst = 0
    for e in errors:
        if e == 1:
            current_burst += 1
            max_burst_length = max(max_burst_length, current_burst)
        else:
            current_burst = 0

    numErrors = sum(errors)

    #burst errors
    burst_count = 0
    for i in range(1, len(errors) - 1):
        if errors[i] == 1 and (errors[i - 1] == 1 or errors[i + 1] == 1):
            burst_count += 1



    plt.figure(figsize=(12, 3))
    plt.bar(range(len(errors)), errors, color='red', edgecolor='black')
    # plt.plot(moving_error_rate, marker='o', markersize=2, linestyle='None', color='blue', label='Moving Error Rate')
    plt.xlabel("Symbol Index")
    plt.title(f"Error Map for Transmitted Symbol '{symbol}' (Across Frames)")
    plt.ylim(-0.1, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)


    for idx, boundary in enumerate(frame_ends):
        plt.axvline(x=boundary - 0.5, color='blue', linestyle='--', linewidth=2, alpha=0.9)
        plt.text(boundary - 0.5, 1.07, f'F{idx+1}',
                 ha='center', va='bottom', fontsize=9,
                 color='blue',
                 bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.2', alpha=0.7))

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
        f"Burst Errors: {burst_count}",
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
    plt.legend(loc='lower left')
    plt.show()

files = getFiles(baseDirPath)
for file in files:
    loadArray(baseDirPath+ "/"+ file)

# sys.exit()
symbols = sortFiles(baseDirPath, files, interpolated=interpolate)
print(symbols["00"][1])
allFrames = []
for i in range(0,5):
    allFrames.append(symbols["01"][i][0])
frameTest = symbols["01"][2][0]

printErrorsWithinFrame(allFrames, "01")

received = getDistribution(symbols["00"][0])

# dists = plotFrameDistribution(symbols, baseDirPath, interpolated=interpolate)
# print(dists["00"][0].mean())
path = "conditional constellation points/single carrier_1744853468/minus_one_minus_jone_rx_scfde_frame0.npy"
# data = loadArray(path)

# data = loadArray("/home/sebastian/LDPC/ldpc/conditionalData/conditional constellation points/single carrier_1744853780/one_minus_jone_rx_scfde_frame1.npy", plot=True)