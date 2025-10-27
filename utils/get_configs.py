import json
import yaml
import sys
import argparse

seq_len_stoi = {
    "1k1k": (1024, 1024),
    "1k8k": (1024, 8192),
    "8k1k": (8192, 1024)
}

def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark matrix from configuration files'
    )
    parser.add_argument(
        '--config-files',
        nargs='+',
        required=True,
        help='One or more configuration files (YAML format)'
    )
    parser.add_argument(
        '--seq-lens',
        choices=list(seq_len_stoi.keys()),
        required=True,
        help=f"Sequence length configuration: {', '.join(seq_len_stoi.keys())}"
    )
    parser.add_argument(
        '--model-prefix',
        required=True,
        help='Model prefix to filter configurations'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help='Step size for concurrency values (default: 2)'
    )
    
    args = parser.parse_args()
    
    isl, osl = seq_len_stoi[args.seq_lens]
    
    all_config_data = {}
    for config_file in args.config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                assert isinstance(config_data, dict), f"Config file '{config_file}' must contain a dictionary"
                
                # Check for duplicate keys, shouldn't really be an issue but with NVIDIA and AMD 
                # separate configs this will help against any possible confusion
                duplicate_keys = set(all_config_data.keys()) & set(config_data.keys())
                if duplicate_keys:
                    raise ValueError(
                        f"Duplicate configuration keys found in '{config_file}': {', '.join(sorted(duplicate_keys))}"
                    )
                
                all_config_data.update(config_data)
        except FileNotFoundError:
            raise ValueError(f"Input file '{config_file}' does not exist.")
    
    matrix_values = []
    for key, val in all_config_data.items():
        # Filter by model prefix i.e., 
        if not key.startswith(args.model_prefix):
            continue

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
                conc *= args.step_size
                if conc > conc_end:
                    conc = conc_end 
    
    print(json.dumps(matrix_values))
    return matrix_values

if __name__ == "__main__":
    main()