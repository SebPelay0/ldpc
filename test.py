import numpy
import matplotlib.pyplot as plt
nums = [73, 97, 71, 26, 8, 42, 57, 95, 99, 95, 70, 14, 5, 66, 100, 88, 91, 52, 81, 19, 25, 54, 86, 71, 8, 49, 27, 53, 11, 50, 78, 90, 86, 49, 52, 76, 35, 56, 28, 62]

print(numpy.mean(nums))
print(numpy.std(nums))
def fiveNumSummary(nums):
    print(f"Min {min(nums)}")
    print(f" Lower Quartile {numpy.quantile(nums, 0.25)}")
    print(f"Median {numpy.median(nums)}")
    print(f"Upper Quartile {numpy.quantile(nums, 0.75)}")
    print(f"Max {max(nums)}")

fiveNumSummary(nums)

def findOutliers(nums, lowerQuartile, upperQuartile):
    iqr = upperQuartile - lowerQuartile
    print(f"Inter-Quartile Range = {iqr}")
    upperBound = upperQuartile + (1.5 * iqr)
    lowerBound = lowerQuartile - (1.5 * iqr)

    print(f"Upper bound {upperBound}")
    print(f"Lower bound {lowerBound}")
    for num in nums:
        if num > upperBound:
            print(f"Upper outlier {num}")
            return True
        if num < lowerBound:
            print(f"Lower outlier {num}")
            return True
    print("No outliers found")
        
findOutliers(nums, 33.25, 82.25)



vals = [861, 1148, 783, 1179, 1083, 896, 977, 1096, 814, 946, 566, 571, 1047, 1073, 982, 1011, 819, 1124, 464, 1174, 947, 770, 626, 1065, 1075, 933, 68, 549, 1138, 912]

plt.plot(vals)
plt.show()