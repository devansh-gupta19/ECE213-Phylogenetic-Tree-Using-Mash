import sys

def count_nodes_pure_python(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the file and strip whitespace/newlines
            nwk_string = file.read().strip()
            
            # In Newick, total nodes = commas + closing parentheses + 1 (for the root)
            comma_count = nwk_string.count(',')
            paren_count = nwk_string.count(')')
            
            total_nodes = comma_count + paren_count + 1
            
            print(f"Total Nodes: {total_nodes}")
            return total_nodes
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Error reading the tree: {e}")

if __name__ == "__main__":
    # Check if the user provided exactly one argument (the script name itself is sys.argv[0])
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_tree_file.nwk>")
    else:
        # sys.argv[1] contains the file path passed by the user
        input_file_path = sys.argv[1]
        count_nodes_pure_python(input_file_path)
