import sys
import os
import math

def split_file(filename, num_parts):
    """
    Split a text file into specified number of equal parts.
    
    Args:
    filename (str): Path to the input file
    num_parts (int): Number of parts to split the file into
    """
    # Validate inputs
    if num_parts < 1:
        print("Number of parts must be at least 1.")
        return

    # Read the entire file content
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return
    except IOError as e:
        print(f"Error reading file: {e}")
        return

    # Calculate the size of each part
    total_length = len(content)
    part_length = math.ceil(total_length / num_parts)

    # Split the content into parts
    for i in range(num_parts):
        # Calculate start and end indices for this part
        start = i * part_length
        end = min((i + 1) * part_length, total_length)

        # Generate output filename
        output_filename = f"{os.path.splitext(filename)[0]}_part{i+1}.txt"

        # Write this part to a new file
        try:
            with open(output_filename, 'w', encoding='utf-8') as out_f:
                out_f.write(content[start:end])
            print(f"Created {output_filename}")
        except IOError as e:
            print(f"Error writing {output_filename}: {e}")

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python split.py <filename> <number_of_parts>")
        sys.exit(1)

    try:
        filename = sys.argv[1]
        num_parts = int(sys.argv[2])
        split_file(filename, num_parts)
    except ValueError:
        print("Number of parts must be an integer.")

if __name__ == "__main__":
    main()
