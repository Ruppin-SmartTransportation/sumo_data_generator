import csv
from termcolor import colored

def compare_csv(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        lines1 = list(reader1)
        lines2 = list(reader2)

        total_lines = max(len(lines1), len(lines2))
        matching_lines = 0

        for i in range(total_lines):
            line1 = lines1[i] if i < len(lines1) else None
            line2 = lines2[i] if i < len(lines2) else None

            if line1 == line2:
                matching_lines += 1
            else:
                print(f"Line {i + 1}:")
                print(colored(f"File1: {line1}", "red") if line1 else colored("File1: [No Line]", "red"))
                print(colored(f"File2: {line2}", "blue") if line2 else colored("File2: [No Line]", "blue"))

        similarity_percentage = (matching_lines / total_lines) * 100 if total_lines > 0 else 0
        print(f"\nSimilarity: {similarity_percentage:.2f}%")

# Usage:
# compare_csv('../export_data/junction_data.csv', '../export_data_from_replay/junction_data.csv')
compare_csv('../export_data/vehicle_data.csv', '../export_data_from_replay/vehicle_data.csv')