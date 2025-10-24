import json
import sys

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "1k8k": (1024, 8192),
    "8k1k": (8192, 1024)
}

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} {{config-file}} {{isl-osl}}")
        exit(1)
        
    config_file = sys.argv[1]
    seq_len = sys.argv[2]
    
    isl, osl = seq_len_stoi.get(seq_len) or (None, None)
    if not (isl or osl):
        raise ValueError(f"Input 'isl-osl' must be one of '{', '.join(seq_len_stoi.keys())}'.")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Input file '{config_file}' does not exist.")

if __name__ == "__main__":
    main()