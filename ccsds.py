
import scipy.io as sio
import ldpc
from pyldpc import coding_matrix
import numpy as np
def readCCSDS(filepath):
    matrixContent = sio.loadmat(filepath)
    H = matrixContent["H"].astype(int)
    G = coding_matrix(np.array(H))
    m, n = H.shape  # Get dimensions of parity-check matrix
    k = n - m  # Number of message bits
    rate = k / n
    print(f"Rate {rate}")
    return H,G

np.random.seed(12)
def testCCSDS():
    test = ldpc.LDPCEncoder(4,8,16, readDataMatrix=True)
    # test.H, test.G = readCCSDS("Matrices/LDPC_CCSDS_256.mat")
    message = np.random.randint(0,2,size=324)
    test.encode(message, -2)
    noisy = test.spreadDSS(4, -2)
    print(f"Noisy {noisy}")
    codeword = test.deSpreadDSS(noisy)
    print(f"Despread {codeword}")
    return test.minSumDecode(codeword)
    
    print(test.H)


testCCSDS()