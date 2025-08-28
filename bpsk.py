
# import numpy
import matplotlib.pyplot as plt
import math
import time
import os
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin")
import scipy
import scipy.io
import sys
import numpy as np
sys.path.append(os.path.abspath("../pyldpc"))
# import pyldpc
from collections import defaultdict
import sdrNonBinary as nb
from loadData import dists
import encoder
import cupy as cp

from cupyx import scatter_add

DT = cp.float32
LLR_CLIP_VAR = DT(30.0)   # like your CPU bitNodes clip
LLR_CLIP_CHK = DT(50.0)   # conservative check->var saturation
EPS_TANH = DT(1e-30)      # to avoid log(0)
MAX_PHI = DT(50.0)        # caps phi_inv arg (ey = exp(-y))
DT = cp.float32
LLR_CLIP_VAR = DT(30.0)   # like your CPU bitNodes clip
LLR_CLIP_CHK = DT(50.0)



DT = cp.float32
EPS_TANH = DT(1e-12)  # for product clipping
TANH_IN_CLIP = DT(20.0)
APOST_CLIP   = DT(30.0)

@cp.fuse()
def phi(x_abs):
    # exact φ(x) = -log(tanh(x/2)), stabilized like CPU tanh(np.clip(...))
    t = cp.tanh(cp.clip(x_abs * DT(0.5), -DT(20.0), DT(20.0)))
    t = cp.maximum(t, DT(1e-30))  # avoid log(0)
    return -cp.log(t)

@cp.fuse()
def phi_inv(y):
    # exact 2*atanh(exp(-y)), stabilized
    y = cp.clip(y, DT(0.0), DT(50.0))
    ey = cp.exp(-y)
    ey = cp.minimum(ey, DT(1.0) - DT(1e-12))
    return cp.log1p(ey) - cp.log1p(-ey)  # == 2*atanh(ey)

def deinterleave(interleaved_bits, depth=13):
    interleaved_bits = np.asarray(interleaved_bits)
    cols = int(np.ceil(len(interleaved_bits) / depth))
    padded_len = cols * depth
    padded_bits = np.pad(interleaved_bits, (0, padded_len - len(interleaved_bits)), constant_values=0)
    matrix = padded_bits.reshape((depth, cols))
    return matrix.T.flatten()[:len(interleaved_bits)]

def computeLLRS(y_complex, distributions):
    re = y_complex.real
    im = y_complex.imag

    p = {}
    for sym in ['00', '01', '10', '11']:
        real_dist = distributions[sym][0]
        imag_dist = distributions[sym][1]
        p[sym] = real_dist.pdf(re) * imag_dist.pdf(im)

    # Small constant to prevent log(0)
    eps = 1e-12

    llr_b0 = np.log((p['00'] + p['01'] + eps) / (p['10'] + p['11'] + eps))
    llr_b1 = np.log((p['00'] + p['10'] + eps) / (p['01'] + p['11'] + eps))

    return [llr_b0, llr_b1]

# numpy.random.seed(29)
# H,G = pyldpc.make_ldpc(8,2, 4,True,True)

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
ORANGE = "\033[38;2;200;100;0m"
RESET = "\033[0m"

# Decoding Iteration 12: BER 0.0014423076923076924
# Decoding Iteration 13: BER 0.0004807692307692308
# Decoding Iteration 14: BER 0.0004807692307692308
# Decoding Iteration 15: BER 0.0004807692307692308
# Decoding Iteration 16: BER 0.0004807692307692308
# Decoding Iteration 17: BER 0.0004807692307692308
# Decoding Iteration 18: BER 0.0004807692307692308
# Decoding Iteration 19: BER 0.0004807692307692308
# Decoding Iteration 20: BER 0.0004807692307692308
FRAME_ERROR = None

class LDPCEncoder():
    def __init__(self, d_v, d_c, n, seed = 20, readDataMatrix= False, matrixPath="Matrices/BG2.mat"):
        self.d_v = d_v # number of times each message bit appears in a parity equation 
        self.d_c = d_c # num bits checked in a parity equation // code rate => 1 -(d_v/d_c)
        self.n = n
        self.seed = seed
        self.PN = None
        if readDataMatrix:
            # H,G = readMatrix("Matrices/parityMatrix.txt")
            H = np.array(readMatrixFile(matrixPath)["H"], dtype=int)
            print(f"H: {H}")
            
            # G = encoder.coding_matrix(H)
            # H= H[:13440, :] # third rate BG2
            # H = H[:10176, :] #Half Rate BG2
            # H = H[:1080,:] #HALF RATE  ROW REMOVAl
            # H = H[:1400,:] #1/3 rate
            # H = H[:1560,:] #1/4 rate
            # scipy.io.savemat("Matrices/5GQuarterRate.mat", {"H":H})
            # H = H[:1400,:] #1/3 rate
            # H = H[:1560,:] #1/4 rate
            if matrixPath == "Matrices/BG2.mat":
                # G = encoder.coding_matrix(H)
                # scipy.io.savemat("Matrices/halfRateBG2.mat", {"G":G})
                print(f"Worker {os.getpid()} is loading G")
                G = np.array(readMatrixFile("Matrices/thirdRateGeneratorBG2.mat")["G"], dtype=int)
            else:
                # input("DOING THIS")
                G = encoder.coding_matrix(H)
            # scipy.io.savemat("Matrices/longGenerator.mat", {"G":G})
            self.H = H
            
            self.G = G
            self.m = H.shape[0]
         
            # numpy.savetxt("H.txt", H, fmt='%d')
            # print(H)
            # print(H.shape)
            # plt.figure(figsize=(10, 4))
            # row_weights = numpy.sum(H, axis=1)
            # column_weights = numpy.sum(H, axis=0)
            # plt.bar(range(len(row_weights)), row_weights)
            # plt.xlabel("Row Index")
            # plt.ylabel("Number of 1s")
            # plt.title("Distribution of 1s in Each Row")
            # plt.show()

            # # Plot column weights
            # plt.figure(figsize=(10, 4))
            # plt.bar(range(len(column_weights)), column_weights)
            # plt.xlabel("Column Index")
            # plt.ylabel("Number of 1s")
            # plt.title("Distribution of 1s in Each Column")
            # plt.show()
            # print(G.shape)
            
        # else:
        #     H,G = pyldpc.make_ldpc(n,d_v, d_c,True,True)
        #     self.H = H
        #     self.G = G
        #     self.m = n * (d_v/d_c) #num check nodesQQ
        

        self.n = H.shape[1]
        self.y = None
        self.originalEncoded = None
        self.SNR = None
        self.bitEnergyRatio = None
        self.numIterations = None
        self.messageDecoded = None
        self.spread = None
        self.BER = 0
        

    def encode(self, message, snr):
        if(len(message) != self.G.shape[1]):
            print("Invalid Message Length: G is " + str(self.G.shape[1]) + " message is " + str(len(message)))
            return
        
        self.originalEncoded = np.dot(self.G, message) % 2
        self.SNR = snr
        noisy = self.originalEncoded
        # noisy = pyldpc.encode(self.G, message, snr)
        # noisy = -1 * self.addNoiseBPSK(snr, self.originalEncoded)
        
        return noisy

    def isValidCodeword(self, decoded_codeword):
        H = cp.asarray(self.H, dtype=cp.int8)
        x = cp.asarray(decoded_codeword, dtype=cp.int8).ravel()
        syndrome = (H.dot(x) & 1).astype(cp.int8)
        return bool(cp.all(syndrome == 0).item())

    # def isValidCodeword(self, decoded_codeword):
    #     syndrome = cp.dot(self.H, decoded_codeword.T) % 2
    #     is_valid = cp.all(syndrome == 0)
    #     # print(f"syndome {syndrome}")
    #     return is_valid

    def bitFlipDecode(self, codeword):
        print(f"       \n H matrix: \n {self.H}")
        bitNodes = []
        #make hard decision on initial bit states
        for val in codeword:
            if(val > 0 ):
                bitNodes.append(0)
            else:
                bitNodes.append(1)
        newMessages = [0] * len(bitNodes)
        numIterations = 0
        
        bitNodeMessages = bitNodes.copy()
        print(f"initial received \n{np.array(bitNodes)}")
        while numIterations < 10:
            if self.isValidCodeword(np.array(bitNodes)):
                errors = np.sum(np.array(self.originalEncoded) != np.array(bitNodes))
                print(f"Bit flip decoding done after {numIterations} iterations")
                return errors/len(bitNodes)

                break
            bitNodeMessages = bitNodes.copy()
            messagesReceivedByBits = [0] * len(bitNodeMessages)
            #Pass the bit node messages into check nodes
            for j in range(0, int(self.m)):
                Ej = np.where(self.H[j] == 1)[0]  #these are the bit nodes that should receive a message from each check node
    
                message = 0
                checkParity = 0
                for i in Ej:
                    checkParity ^= bitNodeMessages[i]  # Compute total check parity

                for target in Ej:
                    parityWithoutBit = checkParity ^ bitNodeMessages[target]  # Remove target bit's effect
                    if parityWithoutBit == 1:
                        messagesReceivedByBits[target] += 1
                    else:
                        messagesReceivedByBits[target] -=1


            for index in range(len(bitNodes)):
                if messagesReceivedByBits[index] > len(Ej) / 2:
                    bitNodes[index] ^= 1  #Flip if majority of checks failed
                    
            print(f"Original code: {self.originalEncoded}")
            print(f"Flipped:       {np.array(bitNodes)}")
            numIterations +=1
         
        errors = np.sum(np.array(self.originalEncoded) != np.array(bitNodes))
        BER = errors/len(bitNodes)
        return BER

    def minSumDecode(self, codeword):
        # codeword = -1  * codeword
        # print(f"Codeword: {self.originalEncoded.flatten()}")
        bitNodes = np.array(codeword, dtype=float)  # Use soft channel values instead of hard bits]
        initialLLRs = []
        numIterations = 0
        # Estimate A Priori LLR's for each bit. 
        for index in range(len(codeword)):
            sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
            if index < 80:
                initialLLRs.append(1e-9)
            else:
                LLR = (2 * float(codeword[index])) / sigma2
                initialLLRs.append(LLR)
        
        # M=> Bit-to-check messages
        M = {}  
        # Initialize bit nodes with the channel LLRs
        for j in range(int(self.m)):
            for i in np.where(self.H[j] == 1)[0]:
            
                if i not in M:
                    M[i] = {}
                M[i][j] = initialLLRs[i]
        
        # for i in range(self.n):
        #     for j in numpy.where(self.H[:, i] == 1)[0]:
        #         M[i][j] = initialLLRs[i]

        #E => Check to bit messages
        E = {j: {} for j in range(int(self.m))}

     

        # # Initialize check-to-bit messages
        
        initialLLRs = np.array(initialLLRs, dtype=float)
        bitNodes = initialLLRs.copy()
        hardDecisions = [0]*len(bitNodes)
        for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
    
        BER = 0
        errors = 0
        while numIterations < 30:
            self.numIterations = numIterations
            errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
          
            print(f"Decoding Iteration {numIterations}: BER {errors/len(bitNodes)}")
            #Test the hard decision on current soft values
            if self.isValidCodeword(np.array(hardDecisions)):
                errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
                print(f"Min Sum Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                self.messageDecoded = hardDecisions
                return errors/len(bitNodes)

            messagesReceivedByBits = np.zeros(len(bitNodes)) 

            for j in range(int(self.m)):  
                Ej = np.where(self.H[j] == 1)[0]  # All bits connected to check j

                if len(Ej) < 2:
                    continue  # Skip underconnected checks

                for target in Ej:
                    # Use all other bits except the target
                    others = [k for k in Ej if k != target]

                    # Get messages Q[k][j] from other bits to this check
                    incoming = [M[k][j] for k in others]
                    magnitudes = np.abs(incoming)
                    signs = np.sign(incoming)
                    signs[signs == 0.0] = 1  # Avoid 0 sign

                    minLLR = np.min(magnitudes)
                    signProd = np.prod(signs)

                    
                    E[j][target] = 0.75 * minLLR * signProd
        
            for i in range(len(bitNodes)):
                incoming_checks = np.where(self.H[:, i] == 1)[0]
                bitNodes[i] = initialLLRs[i] + sum(E[j][i] for j in incoming_checks)
                # bitNodes[i] = numpy.clip(bitNodes[i], -1000.0, 1000.0)
                
            for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
            for i in range(self.n):
                for j in np.where(self.H[:, i] == 1)[0]:
                    other_checks = [k for k in np.where(self.H[:, i] == 1)[0] if k != j]
                    M[i][j] = initialLLRs[i] + sum(E[k][i] for k in other_checks)
            numIterations += 1
        
        
        errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
        BER = errors/len(bitNodes) 
        print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        self.messageDecoded = bitNodes
        self.BER = BER
        
        errors_mask = (np.array(self.originalEncoded) != np.array(hardDecisions)).astype(int)
        error_indices = np.where(errors_mask == 1)[0]
        
        return FRAME_ERROR
    

    def sumProductDecodeTest(self, codeword):

        bitNodes = np.array(codeword, dtype=float)  # Use soft channel valus instead of hard bits]
        print(codeword[:10])
      
        initialLLRs = []
        numIterations = 0
        # Estimate A Priori LLR's for each bit. 
        sigma2 = 1 / (2 *  (10**(self.SNR / 10))) 
        for index in range(len(codeword)):
            # if index < -80:
            #     initialLLRs.append(np.random.uniform(-0.3, 0.3))
            # else:
            LLR = (float(codeword[index])) / sigma2
            initialLLRs.append(LLR)
        
        # M=> Bit-to-check messages
        M = {}  
        # Initialize bit nodes with the channel LLRs
        for j in range(int(self.m)):
            for i in np.where(self.H[j] == 1)[0]:
                if i not in M:
                    M[i] = {}
                M[i][j] = initialLLRs[i]
        
        #E => Check to bit messages
        E = {j: {} for j in range(int(self.m))}

        # # Initialize check-to-bit messages
        
        initialLLRs = np.array(initialLLRs, dtype=float)
        bitNodes = initialLLRs.copy()
        hardDecisions = [0]*len(bitNodes)
        for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
    
        BER = 0
        errors = 0
        while numIterations < 50:
            self.numIterations = numIterations
            errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
            if numIterations < 25:
                print(f"{GREEN}Decoding Iteration {numIterations}: BER {errors/len(bitNodes)} {RESET}")
            if numIterations > 25 and numIterations < 40:
                print(f"{ORANGE}Decoding Iteration {numIterations}: BER {errors/len(bitNodes)} {RESET}")
            if numIterations > 40:
                print(f"{RED}Decoding Iteration {numIterations}: BER {errors/len(bitNodes)} {RESET}")
            #Test the hard decision on current soft values
            if self.isValidCodeword(np.array(hardDecisions)):
                errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
                print(f"Sum Product Decoding done after {numIterations} Iterations, BER {errors/len(bitNodes)}")
                self.messageDecoded = hardDecisions
                return errors/len(bitNodes)

            messagesReceivedByBits = np.zeros(len(bitNodes)) 

            for j in range(int(self.m)):  
                Ej = np.where(self.H[j] == 1)[0]  # All bits connected to check j

                for target in Ej:
                    # Use all other bits except the target
                    others = [k for k in Ej if k != target]

                    # Get valid messages bits to this check
                    incoming = [M[k][j] for k in others]

                    # tanhValues = np.array([np.tanh(M/2) for M in incoming])
                    tanhValues = np.tanh(np.clip(np.array(incoming)/2, -20, 20))

                    tanhProd = np.prod(tanhValues)
                    tanhProd = np.clip(tanhProd, -0.999999999, 0.999999999)

                    E[j][target] =  2*np.arctanh(tanhProd)
                    # E[j][target] = np.clip(2*np.arctanh(tanhProd), -50, 50)

        
                # bitNodes[i] = numpy.clip(bitNodes[i], -100.0, 100.0)
            
            #Set new variable node messages, excluding each check node's own contribution
            # if numIterations < 18:
            #     damping = 0.0
            # elif numIterations < 20:
            #     damping = 0.2
            # elif numIterations < 40:
            #     damping = 0.3
            # else:
            #     damping = 0.6
            damping = 0 
            # damping = 0.3
            for i in range(self.n):
                for j in np.where(self.H[:, i] == 1)[0]:
                    otherChecks = [k for k in np.where(self.H[:, i] == 1)[0] if k != j]
                    newMsg = initialLLRs[i] + sum(E[k][i] for k in otherChecks)
                    M[i][j] = damping * M[i][j] + (1 - damping) * newMsg
                    # M[i][j] =  (initialLLRs[i] + sum(E[k][i] for k in otherChecks))
            
            #Calculate LLR total for the variable node

            for i in range(len(bitNodes)):
                incoming_checks = np.where(self.H[:, i] == 1)[0]
                bitNodes[i] = initialLLRs[i] + sum(E[j][i] for j in incoming_checks)
                
    
            bitNodes = np.clip(bitNodes, -30, 30)
         
            for i in range(len(bitNodes)):
                if bitNodes[i] > 0:
                    hardDecisions[i] = 0
                else: hardDecisions[i] = 1
     
            numIterations += 1
        
        
        errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
        BER = errors/len(bitNodes) 
        print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")

        self.messageDecoded = bitNodes
        self.BER = BER
    
        return FRAME_ERROR
    def _prep_graph_gpu(self):
        """Build device-side edge lists from H once."""
        H = cp.asarray(self.H, dtype=cp.int8)         # (m, n)
        C, N = H.shape
        # Edges are the ones of H
        check_idx, var_idx = cp.where(H == 1)         # device arrays
        # Optionally: sort by check to improve locality (not required)
        order = cp.argsort(check_idx)
        self.edge2check = check_idx[order].astype(cp.int32)   # (E,)
        self.edge2var   = var_idx[order].astype(cp.int32)     # (E,)
        self.num_checks = int(C)
        self.num_vars   = int(N)
        self.num_edges  = int(self.edge2check.size)
    def virtualSumProduct(self, codeword, hardDecisionsIn, useInterleave):
        # --- ensure graph prepared ---
        if not hasattr(self, "edge2check"):
            self._prep_graph_gpu()

        e2c = self.edge2check          # (E,)
        e2v = self.edge2var            # (E,)
        C   = self.num_checks
        N   = self.num_vars

        # --- inputs on device ---
        codeword = cp.asarray(codeword)
        hardDecisions = cp.asarray(hardDecisionsIn, dtype=cp.int8)
        orig_bits = cp.asarray(self.originalEncoded, dtype=cp.int8)

        # --- initial LLRs: compute -> (optional) deinterleave -> puncture ---
        numPuncturedBits = 384
        initialLLRs_host = []
        for symbol in codeword:
            complex_y = complex(*symbol)
            llr_pair = computeLLRS(complex_y, dists)  # must return two LLRs
            initialLLRs_host.extend(llr_pair)

        if useInterleave:
            initialLLRs_host = deinterleave(initialLLRs_host)

        k = min(numPuncturedBits, len(initialLLRs_host))
        initialLLRs_host[:k] = [1e-9] * k

        L_ch = cp.asarray(initialLLRs_host, dtype=cp.float32)
        assert L_ch.size == N, f"Channel LLR length {L_ch.size} != N {N}"

        # --- edge messages ---
        L_v2c = L_ch[e2v].copy()                 # var -> check messages (on edges)
        # (no need to allocate L_c2v separately; compute fresh each iter)
        max_iter = 50

        for it in range(max_iter):
            self.numIterations = it

            # ===================== Check update (CPU-equivalent, deterministic) =====================
            # Use tanh-product with ±20 clamp and product clamp to [-1+1e-12, 1-1e-12]
            # Compute in float64 for numeric parity with NumPy, then cast back to float32.
            Mhalf64      = cp.clip(L_v2c.astype(cp.float64) * 0.5, -20.0, 20.0)
            t64          = cp.tanh(Mhalf64)
            abs_t64      = cp.abs(t64)
            log_abs_t64  = cp.log(cp.clip(abs_t64, 1e-12, 1.0 - 1e-16))
            neg1_i32     = (t64 < 0).astype(cp.int32)

            # Per-check totals via deterministic reductions (no atomics)
            sum_log64    = cp.bincount(e2c, weights=log_abs_t64, minlength=C)                 # float64
            neg_cnt64    = cp.bincount(e2c, weights=neg1_i32.astype(cp.float64), minlength=C) # float64

            # Excluding-self product
            log_excl64   = sum_log64[e2c] - log_abs_t64
            neg_excl32   = (neg_cnt64[e2c].astype(cp.int32) - neg1_i32)
            sign_excl64  = (1 - 2 * (neg_excl32 & 1)).astype(cp.float64)
            abs_excl64   = cp.exp(log_excl64)
            prod_excl64  = sign_excl64 * abs_excl64
            prod_excl64  = cp.clip(prod_excl64, -1.0 + 1e-12, 1.0 - 1e-12)

            L_c2v = (2.0 * cp.arctanh(prod_excl64)).astype(cp.float32, copy=False)
            # (Optionally, keep a very loose clip; CPU did not clip this directly)
            # L_c2v = cp.clip(L_c2v, -50.0, 50.0)

            # ===================== Variable update =====================
            # Sum incoming check messages per variable deterministically
            sum_c2v_per_var = cp.bincount(e2v, weights=L_c2v.astype(cp.float64), minlength=N).astype(cp.float32)

            # A-posteriori LLRs for hard decisions / early stop
            L_post = L_ch + sum_c2v_per_var
            L_post = cp.clip(L_post, -30.0, 30.0)   # matches CPU bitNodes clip

            # Messages var->check exclude self
            L_v2c_new = L_ch[e2v] + (sum_c2v_per_var[e2v] - L_c2v)

            # Damping (your CPU ends up with 0.0; keep it that way for parity)
            # If you want to enable later, insert here.
            L_v2c = cp.clip(L_v2c_new, -40.0, 40.0)  # aligns with tanh(clip(M/2, ±20))

            # ===================== Hard decisions & early stop =====================
            hard = (L_post < 0).astype(cp.int8)
            self.messageDecoded = hard  # maintain side-effect parity with CPU

            # Syndrome (deterministic)
            parity = cp.bincount(e2c, weights=hard[e2v].astype(cp.int32), minlength=C).astype(cp.int32)
            all_ok = bool(cp.all((parity & 1) == 0).get())

            if all_ok:
                errors = int(cp.sum(orig_bits != hard).get())
                ber = errors / N
                self.BER = ber
                input(f"done")
                return ber

            # (Optional debug) — avoid per-iter syncs unless needed
            # if (it & 7) == 0:
            ber_dbg = float(cp.sum(orig_bits != hard).get()) / N
            print(f"[GPU SPA] it={it} BER={ber_dbg:.4g}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")

        # --- fail path (no convergence) ---
        hard = (L_post < 0).astype(cp.int8)
        errors = int(cp.sum(orig_bits != hard).get())
        ber = errors / N
        print(f"Decoding Failed, Best Guess - BER: {ber}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
        self.messageDecoded = hard
        self.BER = ber
        return FRAME_ERROR
    # def virtualSumProduct(self, codeword, hardDecisionsIn, useInterleave):
    #     # --- preconditions / cached graph ---
    #     if not hasattr(self, 'edge2check'):
    #         self._prep_graph_gpu()

    #     E = self.num_edges
    #     N = self.num_vars
    #     C = self.num_checks
    #     e2c = self.edge2check        # (E,)
    #     e2v = self.edge2var          # (E,)

    #     # --- inputs on device ---
    #     codeword = cp.asarray(codeword)
    #     hardDecisions = cp.asarray(hardDecisionsIn).astype(cp.int8)
    #     orig_bits = cp.asarray(self.originalEncoded, dtype=cp.int8)

    #     # --- initial LLRs (keep your logic; vectorize later) ---
    #     # NOTE: this loop can stay for now; the big win is in the message updates below.
    #     numPuncturedBits = 384
    #     initialLLRs_host = []
    #     for symbol in codeword:
    #         complex_y = complex(*symbol)
    #         llr_pair = computeLLRS(complex_y, dists)
    #         initialLLRs_host.extend(llr_pair)
    #     # If you have a deinterleave() function, keep using it to preserve ordering.
    #     if useInterleave:
    #         initialLLRs_host = deinterleave(initialLLRs_host)

    #     initialLLRs_host[:numPuncturedBits] = [1e-9] * numPuncturedBits

    #     L_ch = cp.asarray(initialLLRs_host, dtype=cp.float32)   # (N,)

    #     # --- allocate messages on edges (dicts -> dense arrays) ---
    #     # M[i][j] (var->check)  becomes L_v2c[e] where e is the edge (i,j)
    #     # E[j][i] (check->var)  becomes L_c2v[e]
    #     L_v2c = L_ch[e2v].copy()        # start with channel LLR toward all checks
    #     L_c2v = cp.zeros(E, dtype=cp.float32)
    #     L_post = cp.empty(N, dtype=cp.float32)

    #     # --- iteration (exact SPA, same damping schedule) ---
    #     it = 0
    #     max_iter = 50
    #     while it < max_iter:
    #         self.numIterations = it
    #         # (a) Check-node update: for each check, combine incoming v->c excluding self
    #         #     Exact SPA: sign = parity of negatives; magnitude via φ/φ^-1
    #         neg = (L_v2c < 0).astype(cp.int8)                      # (E,)
    #         abs_v2c = cp.abs(L_v2c)
    #         phi_vals = phi(abs_v2c)                                # (E,)

    #         # per-check totals (sum φ and count negatives)
    #         phi_sum = cp.zeros(C, dtype=cp.float32)
    #         scatter_add(phi_sum, e2c, phi_vals)

    #         neg_cnt = cp.zeros(C, dtype=cp.int32)
    #         scatter_add(neg_cnt, e2c, neg)

    #         # exclude self for each edge (check j = e2c[e])
    #         phi_excl = phi_sum[e2c] - phi_vals
    #         neg_excl = neg_cnt[e2c] - neg
    #         sign_out = 1.0 - 2.0 * (neg_excl & 1).astype(cp.float32)  # 0->+1, 1->-1

    #         L_c2v = sign_out * phi_inv(phi_excl)                     # (E,)
    #         L_c2v = cp.clip(L_c2v, -30.0,30.0)
    #         # (b) Variable-node update: total LLR per var + exclude-self toward each check
    #         sum_c2v_per_var = cp.zeros(N, dtype=cp.float32)
    #         scatter_add(sum_c2v_per_var, e2v, L_c2v)                 # Σ_c' E[c'->v]

    #         # a posteriori LLRs (for hard decisions / early stop)
    #         L_post = L_ch + sum_c2v_per_var
    #         L_post = cp.clip(L_post, -30,30)
    #         # messages v->c = L_ch[v] + Σ_{c'≠c} E[c'->v]
    #         L_v2c_new = L_ch[e2v] + (sum_c2v_per_var[e2v] - L_c2v)   # exclude self

    #         # Damping (exact same schedule/values as your code)
    #         if it < 18:   damping = 0.0
    #         elif it < 20: damping = 0.2
    #         elif it < 40: damping = 0.3
           
    #         damping = 0.0
    #         if damping == 0.0:
    #             L_v2c = L_v2c_new
    #         else:
    #             L_v2c = damping * L_v2c + (1.0 - damping) * L_v2c_new
    #         L_v2c = cp.clip(L_v2c, -40,40)
    #         # (c) Hard decisions + early stop (same parity check as isValidCodeword)
    #         hard = (L_post < 0).astype(cp.int8)
    #         self.messageDecoded = hard
    #         # syndrome: for each check sum the connected bits mod 2
    #         bits_on_edges = hard[e2v]
    #         parity = cp.zeros(C, dtype=cp.int32)
    #         scatter_add(parity, e2c, bits_on_edges)
    #         all_ok = ((parity & 1) == 0).all()

    #         if bool(all_ok):
    #             # Match your return semantics (BER over full frame)
    #             errors = (orig_bits != hard).sum()
    #             return float(errors / N)

    #         it += 1
    #         errors = (orig_bits[:int(N/3)] != hard[:int(N/3)]).sum()
    #         BER = float(errors / N)
    #         print(f" BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")

    #     # --- fail path: match your behavior ---
    #     hard = (L_post < 0).astype(cp.int8)
    #     errors = (orig_bits != hard).sum()
    #     BER = float(errors / N)
    #     print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")
    #     self.messageDecoded = hard
    #     self.BER = BER
    #     return FRAME_ERROR

    # def virtualSumProduct(self, codeword, hardDecisionsIn, useInterleave):
    #     # hardDecisions = deinterleave(hardDecisionsIn.copy())
    #     numPuncturedBits = 384
    #     if useInterleave:
    #         hardDecisions = hardDecisionsIn
    #         print(f"Hard: {hardDecisions[90:100]}")
    #         print(f"Origina: {self.originalEncoded[90:100]}")
    #         bitNodes = np.array(codeword, dtype=float)  # Use soft channel valus instead of hard bits]
    #         # print(codeword[:10])
       
    #         initialLLRs = []
    #         i = 0
    #         for symbol in codeword:
    #             if i < -40:
    #                 initialLLRs.extend([1e-9,1e-9])
    #             else:
    #                 complex_y = complex(*symbol)  # convert (real, imag) tuple to complex number
    #                 llr_pair = computeLLRS(complex_y, dists)
    #                 initialLLRs.extend(llr_pair)
    #             i+=1
    #         initialLLRs = deinterleave(initialLLRs)
    #         initialLLRs[:numPuncturedBits] = [1e-9]* numPuncturedBits #puncture  bits
    #     else:
    #         hardDecisions = hardDecisionsIn
    #         print(f"Hard: {hardDecisions[90:100]}")
    #         print(f"Origina: {self.originalEncoded[90:100]}")
    #         bitNodes = np.array(codeword, dtype=float)  # Use soft channel valus instead of hard bits]
    #         # print(codeword[:10])
       
    #         initialLLRs = []
    #         i = 0
    #         for symbol in codeword:
    #             if i < numPuncturedBits:
    #                 initialLLRs.extend([1e-9,1e-9])
    #             else:
    #                 complex_y = complex(*symbol)  # convert (real, imag) tuple to complex number
    #                 llr_pair = computeLLRS(complex_y, dists)
    #                 initialLLRs.extend(llr_pair)
    #             i+=1

    #     # M=> Bit-to-check messages
    #     M = {}  
    #     # Initialize bit nodes with the channel LLRs
    #     for j in range(int(self.m)):
    #         for i in np.where(self.H[j] == 1)[0]:
    #             if i not in M:
    #                 M[i] = {}
    #             M[i][j] = initialLLRs[i]
       
    #     #E => Check to bit messages
    #     E = {j: {} for j in range(int(self.m))}

    #     # # Initialize check-to-bit messages
       
    #     initialLLRs = np.array(initialLLRs, dtype=float)
    #     bitNodes = initialLLRs.copy()
    #     # hardDecisions = [0]*len(bitNodes)
    #     # for i in range(len(bitNodes)):
    #     #         if bitNodes[i] > 0:
    #     #             hardDecisions[i] = 0
    #     #         else: hardDecisions[i] = 1
   
    #     BER = 0
    #     errors = 0
    #     numIterations = 0
    #     while numIterations < 50:
    #         self.numIterations = numIterations
    #         errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
    #         print(f"Decoding Iteration {numIterations}: BER {errors/len(bitNodes)}")
    #         #Test the hard decision on current soft values
    #         if self.isValidCodeword(np.array(hardDecisions)):
    #             errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
    #             print(f"Sum Product Decoding done after {numIterations} Iterations")
    #             self.messageDecoded = hardDecisions
    #             return errors/len(bitNodes)

    #         messagesReceivedByBits = np.zeros(len(bitNodes))

    #         for j in range(int(self.m)):  
    #             Ej = np.where(self.H[j] == 1)[0]  # All bits connected to check j

    #             for target in Ej:
    #                 # Use all other bits except the target
    #                 others = [k for k in Ej if k != target]

    #                 # Get valid messages bits to this check
    #                 incoming = [M[k][j] for k in others]

    #                 # tanhValues = np.array([np.tanh(M/2) for M in incoming])
    #                 tanhValues = np.tanh(np.clip(np.array(incoming)/2, -20, 20))

    #                 tanhProd = np.prod(tanhValues)
    #                 tanhProd = np.clip(tanhProd, -0.999999999999, 0.999999999999)

    #                 E[j][target] =  2*np.arctanh(tanhProd)
    #                 # E[j][target] = np.clip(2*np.arctanh(tanhProd), -50, 50)

       
    #             # bitNodes[i] = numpy.clip(bitNodes[i], -100.0, 100.0)
           
    #         #Set new variable node messages, excluding each check node's own contribution
    #         if numIterations < 18:
    #             damping = 0.0
    #         elif numIterations < 20:
    #             damping = 0.2
    #         elif numIterations < 40:
    #             damping = 0.3
    #         else:
    #             damping = 0.6
    #         damping = 0
    #         # damping = 0.3
    #         for i in range(self.n):
    #             for j in np.where(self.H[:, i] == 1)[0]:
    #                 otherChecks = [k for k in np.where(self.H[:, i] == 1)[0] if k != j]
    #                 newMsg = initialLLRs[i] + sum(E[k][i] for k in otherChecks)
    #                 M[i][j] = damping * M[i][j] + (1 - damping) * newMsg
    #                 # M[i][j] =  (initialLLRs[i] + sum(E[k][i] for k in otherChecks))
           
    #         #Calculate LLR total for the variable node
    #         for i in range(len(bitNodes)):
    #             incoming_checks = np.where(self.H[:, i] == 1)[0]
    #             bitNodes[i] = initialLLRs[i] + sum(E[j][i] for j in incoming_checks)
               
   
    #         bitNodes = np.clip(bitNodes, -30, 30)
         
    #         for i in range(len(bitNodes)):
    #             if bitNodes[i] > 0:
    #                 hardDecisions[i] = 0
    #             else: hardDecisions[i] = 1
     
    #         numIterations += 1
       
       
    #     errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
    #     BER = errors/len(bitNodes)
    #     print(f"Decoding Failed, Best Guess - BER: {BER}, SNR {self.SNR}, Eb/No {self.bitEnergyRatio}")

    #     self.messageDecoded = hardDecisions
    #     self.BER = BER
   
    #     return FRAME_ERROR


    def addNoiseBPSK(self, SNR_DB, encoded, plot=False):
        power = sum([a**2 for a in encoded]) / len(encoded) 
        
        SNRLinear = 10**(SNR_DB/10)
        noiseStd = np.sqrt(1 / ( SNRLinear*2)  ) 
        noise = noiseStd * np.random.randn(*encoded.shape)
        
        bpsk = 2 * np.array(encoded) - 1  # Convert to -1 and 1

        y = bpsk + noise
        self.SNR = SNR_DB
        self.y = y
        self.bitEnergyRatio = power
        self.symbolNoiseRatio = power
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(range(len(y)), y, 'r.-', label="Noisy Signal (Received Y)")
            plt.plot(range(len(y)), self.originalEncoded, 'bo-', label="Original BPSK Encoded Signal")
            plt.ylabel("Amplitude")
            plt.xlabel("Bit N")
            plt.title(f"BPSK Signal with AWGN (SNR = {SNR_DB} dB)")
            plt.legend()
            plt.grid(True)
            plt.show()
        return y


    def addNoiseQPSK(self, complexSignal, SNR_DB):
        SNRLinear = 10**(SNR_DB/10)
        power = sum([a**2 for a in complexSignal]) / len(complexSignal) #E_b, for BPSK #E_b == E_s
        
   
        noise = np.sqrt(power / (2*SNRLinear)) * (np.random.randn(len(complexSignal)) + 1j *(np.random.randn(len(complexSignal))))
        y = complexSignal + noise
        print(y)
        plt.figure(figsize=(7, 7))
        plt.scatter(y.real, y.imag, color='blue', marker='x', label="Noisy Symbols")
        plt.scatter(complexSignal.real, complexSignal.imag, color='red', marker='o', label="Original QPSK Symbols", s = 35)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlabel("Imaginary")
        plt.ylabel("Real")
        plt.title(f"QPSK Constellation with AWGN (SNR = {SNR_DB} dB)")
        plt.legend()
        plt.show()


    def raw(self, codeword):

        hardDecisions = []
        for val in codeword:
            if(val < 0 ):

                hardDecisions.append(0)
            else:
                hardDecisions.append(1)
        errors = np.sum(np.array(self.originalEncoded) != np.array(hardDecisions))
        return errors/len(codeword)
    
    def generateQPSK(self, length):

        QPSKSymbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        randomSignal = np.random.choice(QPSKSymbols, size=length)
        return randomSignal
    
    def write(self, filePath, SNR, BER, sumProd, bitFlip):
        with open(filePath, "a") as file:
            file.write(f"{time.ctime()}: SNR = {SNR}, MinSum={BER}, SumProd{sumProd}, BitFlip{bitFlip}\n")

    def spreadDSS(self, spreadFactor, snr):
        self.PN = np.random.randint(0,2, size=spreadFactor)
        finalSpread = []
        # print(f"Pseudo-Random Noise Code: {self.PN}")
        for bit in self.originalEncoded:
            result = [A^bit for A in self.PN]
            finalSpread.append(result)
        # print(f"Final {numpy.array(finalSpread)}")
        self.spread = np.array(finalSpread)
        finalSpread = np.array(finalSpread)
        noisy = self.addNoiseBPSK(self.SNR, finalSpread, False)
        return noisy
    
    def deSpreadDSS(self, noisy):
        despreadSoftBits = []

        for spreadBitSequence in noisy:
            xorResult = [bit * (1 if pnBit == 1 else -1) for bit, pnBit in zip(spreadBitSequence, self.PN)]
            softValue = sum(xorResult)  
            
            despreadSoftBits.append(softValue)

        return np.array(despreadSoftBits)



def readMatrix(filePath):
    with open(filePath, "r") as file:
        lines = file.readlines()
    rows = [list(map(int, line.strip())) for line in lines]

    # Convert to NumPy array
    H = np.array(rows, dtype=int)
    # print(H.shape)
    # Generate generator matrix G
    G = pyldpc.coding_matrix(H, False)

    # Save G matrix
    with open("gMatrix.txt", "a") as file:
        file.write(f"G: {G}\n")

    return H, G  # Return both matrices
def readMatrixFile(filePath):
    matrix = scipy.io.loadmat(filePath)
    return matrix


#  Expected Noise Std: 0.7071067811865476, Measured Noise Std: 1.0

# Test = LDPCEncoder(4,5,648, readDataMatrix=True)
def test(snr):
    # DSSS Result
    np.random.seed(12)
    print(f"Starting test SNR = {snr} ")
    message = np.random.randint(0, 2, size=400)
    nonSpread = Test.encode(message, snr)
    
    print(F"Sum Product Result {Test.sumProductDecodeTest(nonSpread)}")
    # print(F"Spread Result {pyldpc.decode(Test.H, sumProdEncode, snr, 30)}")

    #Non-DSSS
    # print(F"Non-Spread Result {Test.minSumDecode(nonSpread)}")
# test(1.2)

# numpy.random.seed(21)
"""RATE 1/2"""
# test0 = LDPCEncoder(2,4, 64)
# message0 = np.random.randint(0, 2, size=541).tolist()  
# # test(10.5)

# """"RATE 3/4"""

# test1 = LDPCEncoder(4,8, 648, readDataMatrix=True)
# test1 = LDPCEncoder(4,5,648, readDataMatrix=True)
# # test1 = LDPCEncoder(2,4, 64, readDataMatrix=False)
# message1 = np.random.randint(0, 2, size=test1.n * 6).tolist()
# path = "Matrices/nonBinaryMatrix.txt"
# H_idx, H_ele, m, n = test1.load_nb_ldpc_matrix(path)
# decoded, nerr = test1.run_nb_ldpc_decoder(message1, H_idx, H_ele, m,n )
# print(nerr)

# """"RATE 1/2"""
# test2 = LDPCEncoder(2, 4, 512)
# message2 = np.random.randint(0, 2, size=519).tolist()  


# """"RATE 1/4"""
# test3 = LDPCEncoder(3,4, 648)
# message3 = np.random.randint(0, 2, size=5).tolist()  


def writeData(filePath, BER, frameError):
    with open(filePath, "a") as file:
        string ="BER Values: ["
        string += ", ".join(str(e) for e in BER) 
        string += "]\n"

        frames = "Frame errors: ["
        frames += ", ".join(str(e) for e in frameError) 
        frames += "]\n"
    

def plot(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    snrRange = np.arange(-4, 4, 1)
    BEROut = []
    
    totalFrameErrors = []
    sumProdBEROut = []
    bitFlipBEROut = []
    n = 4
    for snr in snrRange:
        avgBER = 0
        avgSumProdBER = 0
        avgBitFlipBER = 0
        frameErrors = 0
        for it in range(n):
            sumProdBER = 0
            BER = 0
            bitFlipBER = 0

            if readMatrixFile:
                test1.H, test1.G = readMatrix("parityMatrix.txt")
            if minSum:
                message1 = np.random.randint(0, 2, size=327).tolist()  
                test1.encode(message1) 
                noisyCodeword = test1.addNoiseBPSK(snr)
                # BER = test0.bitFlipDecode(noisyCodeword)
                BER = test1.minSumDecode(noisyCodeword)
            
            if sumProd:
                sumProdEncode = pyldpc.encode(test1.G, message1, snr)
                sumProdDecode = pyldpc.decode(test1.H, sumProdEncode, snr, 30) #maybe change this to use var snr
                sumProdMessage = pyldpc.get_message(test1.G, sumProdDecode)
                sumProdErrors = np.sum(np.array(message1) != np.array(sumProdMessage))
                sumProdBER = sumProdErrors/len(sumProdMessage)  

            if bitFlip:
                message1 = np.random.randint(0, 2, size=325).tolist()  
                test1.encode(message1) 
                noisyCodeword = test1.addNoiseBPSK(snr)
                bitFlipBER = 1 - test1.bitFlipDecode(noisyCodeword)
            
            if BER > 0:
                frameErrors +=1
            avgBER += BER
            avgSumProdBER += sumProdBER
            avgBitFlipBER += bitFlipBER
            # if BER == 0:
            #     break
            
        BEROut.append(avgBER/n)
        sumProdBEROut.append(avgSumProdBER/n)
        bitFlipBEROut.append(avgBitFlipBER/n)
        totalFrameErrors.append(frameErrors/n)
        test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
       

    writeData("data.txt", BEROut, totalFrameErrors)
   
    plt.figure(figsize=(8, 5))
    plt.yscale("log")  # Set the Y-axis to logarithmic scale

    plt.plot(snrRange, BEROut, 'b-', label="Min-Sum Decoding")
    plt.plot(snrRange, sumProdBEROut, 'r-', linestyle='-', label=" Sum-Prod Decoding")
    plt.plot(snrRange, bitFlipBEROut, 'g-', linestyle='-', label=" Bit-Flip Decoding")
   
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Decoding: BER vs. SNR at 1/2 Data Rate (n=648, BPSK)")

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frame Error Rate")
    plt.title("LDPC Bit-Flip Decoding: Frame Error vs. SNR at 1/4 Data Rate, n= 256, BPSK")
    plt.grid(True, which="both", linestyle="--")
    plt.show()

    

def plotRates():
    snrRange = np.arange(-3.5, -3, -2.5,-2)
    BEROut = []
    
    highBEROut = []
    threeQuarterBEROut = []
    halfBEROut = []
    quarterBEROut = []
  
    n = 10
    for snr in snrRange:
        highAvg = 0
        threeQuarterAvg = 0
        halfAverage = 0
        quarterAverage = 0
        for it in range(n):
            ##High Rate
            message0 = np.random.randint(0, 2, size=541).tolist()  
            test0.encode(message0)
            highBER = test0.minSumDecode(test0.addNoiseBPSK(snr))
            highAvg += highBER

            # ##Three Quarter
            message1 = np.random.randint(0, 2, size=487).tolist()  
            test1.encode(message1)
            threeQuarterBer = test1.minSumDecode(test1.addNoiseBPSK(snr))
            threeQuarterAvg += threeQuarterBer

            ##Half 
            message2 = np.random.randint(0, 2, size=257).tolist()  
            test2.encode(message2)
            halfBER = test2.minSumDecode(test2.addNoiseBPSK(snr))
            halfAverage += halfBER

            ##Quarter
            message3 = np.random.randint(0, 2, size=164).tolist()  
            test3.encode(message3)
            quarterBER = test3.minSumDecode(test3.addNoiseBPSK(snr))
            quarterAverage += quarterBER

        highBEROut.append(highAvg/n)
        threeQuarterBEROut.append(threeQuarterAvg/n)
        halfBEROut.append(halfAverage/n)
        quarterBEROut.append(quarterAverage/n)



    plt.figure(figsize=(8, 5))
    plt.yscale("log") 

    plt.plot(snrRange, halfBEROut, 'r-', label="R = 1/2")


    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Decoding: BER vs. SNR at 1/2 Data Rate (n=512, BPSK)")

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()

def plotFrameError(minSum=True, sumProd=False, bitFlip=False, readMatrixFile=False):
    print("Begin frame error plot")
    snrRange = np.array([2.6])

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
            os.system("clear")
            print(f"Iteration No. {iterations}, SNR: {snr}, Frame Errors: {frameErrors}, FER {frameErrors/iterations}")
            print(f"SNR RANGE: {snrRange}")
            message1 = np.random.randint(0, 2, size=324).tolist()  
            
            noist = test1.encode(message1, snr)
            # noisy = test1.spreadDSS(4, snr)
            # codeword = test1.deSpreadDSS(noisy)
            BER =  test1.sumProductDecodeTest(noist)
            if BER != FRAME_ERROR:
                BERS.append(BER)

            # BER = test1.minSumDecode(pyldpc.encode(test1.G, message1, snr))
            if BER is  FRAME_ERROR:
                BERS.append(1)
                frameErrors += 1
            if iterations > 100000 or frameErrors > 30:
                # frameErrors = 0
                break
            # if iterations == 200 and frameErrors == 0:
            #     break
        
        totalFrameErrors.append(frameErrors/iterations)


        # test1.write("results2.txt", snr, avgBER/n, avgSumProdBER/5.5,avgBitFlipBER/n )
    print(f"SNRS: {snrRange}")
    print(f"Total frame errors: {totalFrameErrors}")
    print(f"SNR: -2, FER = 0.0990990990990991")
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrRange, totalFrameErrors, marker='o', linestyle='-')  
    plt.xlabel("SNR (dB)")
    
    plt.ylabel("Frame Error Rate")
    plt.title("Sum Product 5G LDPC Frame Error vs. SNR at 1/5 Data Rate, n= 2000, z = 80")
    plt.grid(True, which="both", linestyle="--")
    
    plt.show() 
# plotFrameError()

# plotFrameError()