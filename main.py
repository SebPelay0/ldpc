from bpsk import LDPCEncoder
import numpy as np
import os
def main():
    print("Enter simulation details...\n")

    while(input("Exit? (Y/N)") != "Y"):
        try:
            existing = input("Load an existing matrix (.mat) file? Y/N\n");
            if existing == "Y":
                path = input("Enter filepath to matrix\n")
                Test = LDPCEncoder(4,5,2000, readDataMatrix=True)
                selectDecoder = input("For min-sum decoding enter: A \nFor sum-product enter: B\n")
                os.system("cls")
                singleTest = input("Execute a single test? (Y/N)\n")
                if singleTest == "Y":
                    snr = int(input("Select test SNR (DB)\n"))
                    os.system("cls")
                    message = np.random.randint(0, 2, size=400)
                    nonSpread = Test.encode(message, snr)
                    print(F"Sum Product Result {Test.sumProductDecodeTest(nonSpread)}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()