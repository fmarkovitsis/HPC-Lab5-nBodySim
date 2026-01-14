import subprocess
import re
import statistics
import sys

# --- CONFIGURATION ---
# Replace this with the actual command/path to your compiled CUDA executable
EXECUTABLE_PATH = ["./nbody"] 

ITERATIONS = 12

def parse_output(output_string):
    """
    Parses the stdout from the CUDA program to find Time and Throughput.
    """
    # Looks for floating point numbers immediately preceding "seconds"
    time_match = re.search(r"Total Comp Time:\s+([0-9]*\.?[0-9]+)\s+seconds", output_string)
    
    # Regex for "Average Throughput: "throughput" Billion Interactions / second"
    throughput_match = re.search(r"Average Throughput:\s+([0-9]*\.?[0-9]+)\s+Billion", output_string)

    if time_match and throughput_match:
        return float(time_match.group(1)), float(throughput_match.group(1))
    else:
        return None, None

def calculate_stats(data_list, label):
    """
    Removes the min and max values, then calculates mean and stdev.
    """
    if len(data_list) < 3:
        print(f"Error: Not enough data points to remove outliers for {label}.")
        return

    sorted_data = sorted(data_list)
    
    # Remove the smallest (index 0) and biggest (index -1)
    trimmed_data = sorted_data[1:-1]
    mean_val = statistics.mean(trimmed_data)
    stdev_val = statistics.stdev(trimmed_data)
    
    return mean_val, stdev_val

def main():
    times = []
    throughputs = []

    print(f"Starting benchmark: Running {ITERATIONS} iterations...\n")
    print("-" * 60)

    for i in range(1, ITERATIONS + 1):
        try:
            # Run the CUDA executable
            result = subprocess.run(
                EXECUTABLE_PATH, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            t, tp = parse_output(result.stdout)
            
            if t is not None and tp is not None:
                times.append(t)
                throughputs.append(tp)
                print(f"Run {i:02d}: Time = {t:.4f} s | Throughput = {tp:.4f} Billion/s")
            else:
                print(f"Run {i:02d}: Failed to parse output.")
                
        except subprocess.CalledProcessError as e:
            print(f"Run {i:02d}: Error - Executable failed with return code {e.returncode}")
        except FileNotFoundError:
            print(f"Error: Could not find executable at {EXECUTABLE_PATH[0]}")
            sys.exit(1)

    print("-" * 60)
    
    if len(times) != ITERATIONS:
        print(f"Warning: Only collected {len(times)} valid data points out of {ITERATIONS}.")

    # --- Process Time ---
    time_mean, time_std = calculate_stats(times, "Time")
    
    # --- Process Throughput ---
    tp_mean, tp_std = calculate_stats(throughputs, "Throughput")

    # --- Final Report ---
    print("\nBenchmark Results (Outliers Removed):")
    print("=" * 40)
    print(f"Total Computation Time:")
    print(f"  Mean:  {time_mean:.6f} seconds")
    print(f"  Stdev: {time_std:.6f}")
    print("-" * 40)
    print(f"Average Throughput:")
    print(f"  Mean:  {tp_mean:.6f} Billion Interactions/sec")
    print(f"  Stdev: {tp_std:.6f}")
    print("=" * 40)

if __name__ == "__main__":
    main()