import matplotlib.pyplot as plt
import numpy as np
import math
# Constants
waveSpeed = 1500  
relativeSpeed = 10  
timeSteps = np.linspace(1, 100, num=1000)


def quadratic(x):
    return math.exp(x) + 12000  


def dopplerScale(freq, relativeSpeed):
    scaleFactor = 1 - relativeSpeed / waveSpeed
    return scaleFactor * freq


frequencies = np.array([quadratic(t) for t in timeSteps])
firstPath = np.array([dopplerScale(f, relativeSpeed) for f in frequencies])


differences = frequencies - firstPath


scaledDifferences = differences   # Increased scaling factor


percentageShift = (differences / frequencies) * 100  # Convert to %


plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(timeSteps, frequencies, label="Original Frequency", color="blue")
plt.plot(timeSteps, firstPath, label="Doppler-Shifted Frequency", color="green", linestyle="dashed")
plt.ylabel("Frequency (Hz)")
plt.title("Doppler Shift in Frequency Over Time")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(timeSteps, scaledDifferences, label="Frequency Shift (Scaled x10^8)", color="red")
plt.plot(timeSteps, percentageShift, label="Relative Frequency Shift (%)", color="purple")
plt.xlabel("Time Steps")
plt.ylabel("Frequency Shift")
plt.yscale("log")  # **Apply Log Scale**
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
