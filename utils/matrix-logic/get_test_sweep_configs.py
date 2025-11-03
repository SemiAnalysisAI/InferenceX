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
        description='Generate benchmark matrix from a specific configuration key'
    )
    parser.add_argument(
        '--config-files',
        nargs='+',
        required=True,
        help='One or more configuration files (YAML format)'
    )
    parser.add_argument(
        '--key',
        required=True,
        help='Configuration key to use'
    )
    parser.add_argument(
        '--seq-lens',
        nargs='+',
        choices=list(seq_len_stoi.keys()),
        required=False,
        help=f"Sequence length configurations to include: {', '.join(seq_len_stoi.keys())}. If not specified, all sequence lengths are included."
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=2,
        help='Step size for concurrency values (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Convert seq-lens to set of (isl, osl) tuples for filtering
    seq_lens_filter = None
    if args.seq_lens:
        seq_lens_filter = {seq_len_stoi[sl] for sl in args.seq_lens}
    
    # Load and merge all config files
    all_config_data = {}
    for config_file in args.config_files:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                assert isinstance(config_data, dict), f"Config file '{config_file}' must contain a dictionary"
                
                # Check for duplicate keys
                duplicate_keys = set(all_config_data.keys()) & set(config_data.keys())
                if duplicate_keys:
                    raise ValueError(
                        f"Duplicate configuration keys found in '{config_file}': {', '.join(sorted(duplicate_keys))}"
                    )
                
                all_config_data.update(config_data)
        except FileNotFoundError:
            raise ValueError(f"Input file '{config_file}' does not exist.")
    
    # Check if the key exists
    if args.key not in all_config_data:
        available_keys = ', '.join(sorted(all_config_data.keys()))
        raise ValueError(
            f"Key '{args.key}' not found in configuration files. "
            f"Available keys: {available_keys}"
        )
    
    val = all_config_data[args.key]
    
    # Validate required fields
    seq_len_configs = val.get('seq-len-configs')
    assert seq_len_configs, f"Missing 'seq-len-configs' for key '{args.key}'"
    
    image = val.get('image')
    model = val.get('model')
    precision = val.get('precision')
    framework = val.get('framework')
    runner = val.get('runner')
    
    assert None not in (image, model, precision, framework, runner), \
        f"Missing required fields (image, model, precision, framework, runner) for key '{args.key}'"
    
    matrix_values = []
    
    # Process each sequence length configuration
    for seq_config in seq_len_configs:
        isl = seq_config.get('isl')
        osl = seq_config.get('osl')
        
        assert None not in (isl, osl), \
            f"Missing 'isl' or 'osl' in seq-len-config for key '{args.key}'"
        
        # Filter by sequence lengths if specified
        if seq_lens_filter and (isl, osl) not in seq_lens_filter:
            continue
        
        bmk_space = seq_config.get('bmk-space')
        assert bmk_space, f"Missing 'bmk-space' in seq-len-config for key '{args.key}'"
        
        for bmk in bmk_space:
            tp = bmk.get('tp')
            conc_start = bmk.get('conc-start')
            conc_end = bmk.get('conc-end')
            ep = bmk.get('ep')
            dp_attn = bmk.get('dp-attn')
            
            assert None not in (tp, conc_start, conc_end), \
                f"Missing 'tp', 'conc-start', or 'conc-end' in bmk-space for key '{args.key}'"
            
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
                    'conc': conc,
                    'max-model-len': isl + osl,
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