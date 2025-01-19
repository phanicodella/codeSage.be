# Script to extract variable names before '=' from a .env file in the same directory

def extract_vars_from_env():
    file_path = ".env"
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        vars_before_equal = []

        for line in lines:
            # Ignore empty lines or lines without '='
            if '=' in line:
                # Split the line at the '=' sign and get the part before it
                var_name = line.split('=', 1)[0].strip()
                vars_before_equal.append(var_name)

        return vars_before_equal

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found in the current directory.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    vars_list = extract_vars_from_env()

    if vars_list:
        print("\nVariables before '=' in the .env file:")
        for var in vars_list:
            print(var)
    else:
        print("No variables found or an error occurred.")
