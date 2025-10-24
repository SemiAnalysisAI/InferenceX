import json
import sys

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "1k8k": (1024, 8192),
    "8k1k": (8192, 1024)
}

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} {{config-file}} {{isl-osl}} [step-size]")
        exit(1)
        
    config_file = sys.argv[1]
    seq_len = sys.argv[2]
    step_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    isl, osl = seq_len_stoi.get(seq_len) or (None, None)
    if not (isl or osl):
        raise ValueError(f"Input 'isl-osl' must be one of '{', '.join(seq_len_stoi.keys())}'.")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            assert isinstance(config_data, dict)
    except FileNotFoundError:
        raise ValueError(f"Input file '{config_file}' does not exist.")
    
    matrix_values = []
    for key, val in config_data.items():
        seq_len_configs = val.get('seq-len-configs')
        assert seq_len_configs, f"Missing 'seq-len-configs' for key '{key}'"
        
        image = val.get('image')
        model = val.get('model')
        precision = val.get('precision')
        framework = val.get('framework')
        runner = val.get('runner')
        
        assert None not in (image, model, precision, framework, runner), \
            f"Missing required fields for key '{key}'"
        
        # Check if this config has matching sequence lengths
        matching_seq_config = None
        for slq in seq_len_configs:
            if slq.get('isl') == isl and slq.get('osl') == osl:
                matching_seq_config = slq
                break
        
        if not matching_seq_config:
            continue  # Skip this config if no matching sequence length, this is possible
        
        bmk_space = matching_seq_config.get('bmk-space')
        assert bmk_space, f"Missing 'bmk-space' in matching seq-len-config for key '{key}'"
        
        for bmk in bmk_space:
            tp = bmk.get('tp')
            conc_start = bmk.get('conc-start')
            conc_end = bmk.get('conc-end')
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')
            
            assert None not in (tp, conc_start, conc_end), \
                f"Missing 'tp', 'conc-start', or 'conc-end' in bmk-space for key '{key}'"
            
            # Generate entries for each concurrency value in the range
            conc = conc_start
            while conc <= conc_end:
                entry = {
                    'image': image,
                    'model': model,
                    'precision': precision,
                    'framework': framework,
                    'runner': runner,
                    'isl': isl,
                    'osl': osl,
                    'tp': tp,
                    'conc': conc
                }
                
                # Add optional fields if they exist
                if ep is not None:
                    entry['ep'] = ep
                if dp_attn is not None:
                    entry['dp-attn'] = dp_attn
                
                matrix_values.append(entry)
                
                if conc == conc_end:
                    break
                conc *= step_size
                if conc > conc_end:
                    conc = conc_end 
    
    print(json.dumps(matrix_values))
    return matrix_values

if __name__ == "__main__":
    main()