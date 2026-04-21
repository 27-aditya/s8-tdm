"""
run_all.py
Master script to execute Part 1 (Single-Objective) and Part 2 (Multi-Objective).
"""
import time
import os
from part1_clustering import run_part1
from part2_multiobjective import run_part2

def main():
    start_time = time.time()
    
    print("="*65)
    print("STARTING ASSIGNMENT II: DATA MINING CLUSTERING PROJECT")
    print("="*65)

    # 1. Run Part 1: K-Means, DBSCAN, and Statistical Tests
    print("\n>>> EXECUTING PART 1...")
    run_part1()

    # 2. Run Part 2: Multi-Objective Clustering (NSGA-II)
    print("\n>>> EXECUTING PART 2...")
    run_part2()

    end_time = time.time()
    print("\n" + "="*65)
    print(f"PROJECT EXECUTION COMPLETE in {end_time - start_time:.2f} seconds")
    print("Check your folder for generated .png plots and statistical results.")
    print("="*65)

if __name__ == "__main__":
    main()