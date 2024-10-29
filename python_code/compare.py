def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    diff = set(lines1).difference(lines2)

    if diff:
        print("Differences found:")
        for line in diff:
            print(line.strip())
    else:
        print("No differences found.")

# Usage example
file1 = '../build/out/combined_output_C1.dat'
file2 = '../build/serial_out/output_C1_0.dat'
compare_files(file1, file2)