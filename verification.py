#######################################################
# Verification for GPU results of N-Body simulation   #
# Garyfallia Anastasia Papadouli | 03533              #                        
# Filippos Markovitsis |                              #
# Israel Sanchez Cabrera |                            #                    
#######################################################

import struct
import sys
import math

FILE_CPU = "final_CPU.bin"
FILE_GPU = "final_GPU.bin"
TOLERANCE = 1e-3 #0.001
def compare_files(file1, file2, tol):
    print(f"\x1b[31mComparing {file1} and {file2} with tolerance {tol}...\033[0m")

    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            header_size = struct.calcsize('ii') #size of the 2 parts of header
            raw_h1 = f1.read(header_size)
            raw_h2 = f2.read(header_size)

            if not raw_h1 or not raw_h2:
                print("File empty/missing header")
                return False
            n_sys1, n_body1 = struct.unpack('ii', raw_h1)
            n_sys2, n_body2 = struct.unpack('ii', raw_h2)
            if n_sys1 != n_sys2 or n_body1 != n_body2:
                print("HEADER MISMATCH:")
                print(f"CPU File: {n_sys1} systems, {n_body1} bodies")
                print(f"GPU FIle: {n_sys2} systems, {n_body2} bodies")
                return False
            print("Headers Match.")
            print(f"Header: {n_sys1} systems, {n_body1} bodies")
            body_fmt = 'ffffff' # x,y,z,vx,vy,vz (24 bytes)
            body_size = struct.calcsize(body_fmt)
            total_bodies = n_sys1 * n_body1

            for i in range(total_bodies):
                # Read binary chunks
                b1_raw = f1.read(body_size)
                b2_raw = f2.read(body_size)

                # Check for unexpected EOF
                if len(b1_raw) != body_size or len(b2_raw) != body_size:
                    print(f"Error: Unexpected End of File at body index {i}")
                    return False
                
                v1 = struct.unpack(body_fmt, b1_raw)
                v2 = struct.unpack(body_fmt, b2_raw)

                labels = ['x', 'y', 'z', 'vx', 'vy', 'vz']
                for k in range(6):
                    diff = abs(v1[k] - v2[k])
                    
                    if diff > tol:
                        system_idx = i // n_body1
                        body_idx = i % n_body1  
                        print(f"\033[31m\nMISMATCH FOUND!")
                        print(f"Location: System {system_idx}, Body {body_idx}")
                        print(f"Component: {labels[k]}")
                        print(f"File A Value: {v1[k]:.6f}")
                        print(f"File B Value: {v2[k]:.6f}")
                        print(f"Difference:   {diff:.6f}")
                        print("Stopping comparison.\033[0m")
                        return False # Return point on failure
        print("\nSUCCESS: All bodies match within tolerance!")
        return True

    except FileNotFoundError as e:
        print(f"File error: {e}")
        return False

if __name__ == "__main__":
    compare_files(FILE_CPU, FILE_GPU, TOLERANCE)