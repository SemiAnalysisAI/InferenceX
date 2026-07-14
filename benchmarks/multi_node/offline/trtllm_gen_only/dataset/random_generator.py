import random
import json
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import numpy as np
import tqdm
import logging
import argparse
from transformers import AutoTokenizer, AutoConfig
from multiprocessing import Pool, cpu_count
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_chat_template_kwargs(custom_tokenizer):
    """Build apply_chat_template kwargs. deepseek_v4 always runs in Think Max mode."""
    if custom_tokenizer == "deepseek_v4":
        return {"enable_thinking": True, "reasoning_effort": "max"}
    return {}


random.seed(42)
np.random.seed(42)


def get_chat_template_token_number(tokenizer, chat_template_kwargs=None):
    chat_template_kwargs = chat_template_kwargs or {}
    message = tokenizer.apply_chat_template(
        [{"role": "user", "content": "a"}],
        add_generation_prompt=True,
        tokenize=False,
        **chat_template_kwargs,
    )
    encode_ids = tokenizer.encode(message, add_special_tokens=False)
    return len(encode_ids) - 1


def _create_random_text_and_tokens(tokenizer, vocab_size, input_len, output_len, num_prompts, random_ratio, chat_template_kwargs=None):
    """Helper function to create random text and calculate token count."""
    chat_template_kwargs = chat_template_kwargs or {}
    chat_template_token_number = get_chat_template_token_number(tokenizer, chat_template_kwargs)
    print(f"chat_template_token_number: {chat_template_token_number}")
    input_len = input_len - chat_template_token_number

    input_lens = np.random.randint(
        int(input_len * random_ratio) if input_len > 1 else 1,
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * random_ratio) if output_len > 1 else 1,
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, vocab_size, size=num_prompts)

    prompts = []

    # origin_ids = []
    # for i in range(num_prompts):
    #     origin_id = [(offsets[i] + i + j) %
    #                  tokenizer.vocab_size for j in range(int(input_lens[i]*1.5))]
    #     origin_ids.append(origin_id)
    # print(f"length of origin_ids: {len(origin_ids)}")

    # origin_texts = tokenizer.batch_decode(origin_ids)
    # print(f"length of origin_texts: {len(origin_texts)}")
    # re_encoded_sequences = tokenizer.batch_encode_plus(
    #     origin_texts, add_special_tokens=False)
    # print(f"length of re_encoded_sequences: {len(re_encoded_sequences)}")
    # print(f"length of input_ids: {len(re_encoded_sequences['input_ids'])}")

    # re_encoded_ids = []
    # for i in range(num_prompts):
    #     re_encoded_ids.append(
    #         # 使用 'input_ids' 键
    #         re_encoded_sequences['input_ids'][i][:input_lens[i]])
    # print(f"length of re_encoded_ids: {len(re_encoded_ids)}")

    # re_encoded_texts = tokenizer.batch_decode(re_encoded_ids)
    # print(f"length of re_encoded_texts: {len(re_encoded_texts)}")

    # for i in range(num_prompts):
    #     prompt = tokenizer.apply_chat_template(
    #         [{"role": "user", "content": re_encoded_texts[i]}],
    #         add_generation_prompt=True,
    #         tokenize=False,
    #     )
    #     input_lens[i] += chat_template_token_number
    #     prompts.append(prompt)

    for i in range(num_prompts):
        origin_text = tokenizer.decode([(offsets[i] + i + j) %
                                        vocab_size for j in range(int(input_lens[i]*1.5))])
        re_encoded_sequence = tokenizer.encode(origin_text, add_special_tokens=False)[
            :(input_lens[i])
        ]
        prompt_text = tokenizer.decode(re_encoded_sequence)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
            **chat_template_kwargs,
        )
        input_lens[i] += chat_template_token_number
        prompts.append(prompt)

    return prompts, input_lens, output_lens


def _random_generate_worker(args):
    """Worker function for parallel processing"""
    (worker_id, num_prompts, num_tokens, max_tokens, tokenizer_name, vocab_size,
     random_ratio, custom_tokenizer) = args

    # Initialize tokenizer in worker process
    if custom_tokenizer and custom_tokenizer == "deepseek_v32":
        from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer
        tokenizer = DeepseekV32Tokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True)
    elif custom_tokenizer and custom_tokenizer == "deepseek_v4":
        from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
        tokenizer = DeepseekV4Tokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True)

    chat_template_kwargs = _build_chat_template_kwargs(custom_tokenizer)
    prompts, input_lens, output_lens = _create_random_text_and_tokens(
        tokenizer, vocab_size, num_tokens, max_tokens, num_prompts, random_ratio,
        chat_template_kwargs)
    results = []
    for i in range(num_prompts):
        prompt = prompts[i]
        input_len = input_lens[i]
        output_len = output_lens[i]
        prompt_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "num_tokens": int(input_len),  # Convert numpy.int64 to Python int
            "expected_tokens": num_tokens,      # Record expected token count
            "max_tokens": int(output_len),  # Convert numpy.int64 to Python int
        }
        results.append(prompt_data)
    return results


class RandomPromptGenerator:
    """
    Random prompt generator using random token IDs
    Based on the structure of OptimizedPromptGenerator
    """

    def __init__(self, tokenizer_name: str = "deepseek-ai/DeepSeek-R1", custom_tokenizer: str = None, vocab_from_config: bool = False):
        """Initialize with tokenizer"""
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        if custom_tokenizer and custom_tokenizer == "deepseek_v32":
            from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer
            self.tokenizer = DeepseekV32Tokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True)
        elif custom_tokenizer and custom_tokenizer == "deepseek_v4":
            from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
            self.tokenizer = DeepseekV4Tokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True)
        if vocab_from_config:
            self.config = AutoConfig.from_pretrained(
                tokenizer_name, trust_remote_code=True)
            self.vocab_size = self.config.vocab_size
        else:
            try:
                self.vocab_size = self.tokenizer.vocab_size
            except (AttributeError, NotImplementedError):
                # Custom tokenizers wrap an inner HF tokenizer
                self.vocab_size = self.tokenizer.tokenizer.vocab_size
        self.tokenizer_name = tokenizer_name
        self.custom_tokenizer = custom_tokenizer
        self.chat_template_kwargs = _build_chat_template_kwargs(custom_tokenizer)
        logger.info(f"Tokenizer loaded. Vocabulary size: {self.vocab_size}")

    def generate_prompts_batch(
        self,
        num_prompts: int,
        num_tokens: int = 1024,
        max_tokens: int = 1024,
        random_ratio: float = 1.0
    ) -> List[Dict]:
        """Generate multiple random prompts in batch (single process)"""
        logger.info(
            f"Generating {num_prompts} random prompts with target {num_tokens} tokens each (single process)")

        # 修改变量名，避免与函数参数冲突
        prompt_texts, input_lens, output_lens = _create_random_text_and_tokens(
            self.tokenizer, self.vocab_size, num_tokens, max_tokens, num_prompts, random_ratio,
            self.chat_template_kwargs)

        results = []  # 使用 results 而不是 prompts
        for i in tqdm.tqdm(range(num_prompts), desc="Generating random prompts"):
            prompt_text = prompt_texts[i]  # 使用 prompt_texts
            input_len = input_lens[i]
            output_len = output_lens[i]

            prompt_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text},  # 使用 prompt_text
                ],
                "num_tokens": int(input_len),
                "expected_tokens": num_tokens,
                "max_tokens": int(output_len),
            }
            results.append(prompt_data)  # 使用 results
        return results

    def generate_prompts_parallel(
        self,
        num_prompts: int,
        num_tokens: int = 1024,
        max_tokens: int = 1024,
        num_workers: Optional[int] = None,
        random_ratio: float = 1.0
    ) -> List[Dict]:
        """
        Generate multiple random prompts using parallel processing
        """
        if num_workers is None:
            num_workers = min(cpu_count(), 8)  # Limit to 8 workers max

        logger.info(
            f"Generating {num_prompts} random prompts with target {num_tokens} tokens each")
        logger.info(f"Using {num_workers} workers for parallel processing")

        # Prepare work distribution
        prompts_per_worker = num_prompts // num_workers
        remaining = num_prompts % num_workers

        worker_args = []
        for i in range(num_workers):
            num_prompts_for_worker = prompts_per_worker + \
                (1 if i < remaining else 0)
            worker_args.append(
                (i, num_prompts_for_worker, num_tokens, max_tokens, self.tokenizer_name,
                 self.vocab_size, random_ratio, self.custom_tokenizer))

        # Execute parallel processing
        all_prompts = []
        with Pool(num_workers) as pool:
            results = list(tqdm.tqdm(
                pool.imap(_random_generate_worker, worker_args),
                total=num_workers,
                desc="Generating prompts",
                unit="worker",
                postfix={"total_prompts": num_prompts}
            ))

            for worker_results in results:
                all_prompts.extend(worker_results)
        return all_prompts

    def dump_to_file(
        self,
        prompts: List[Dict],
        output_file: str,
        format_type: str = "serve"
    ):
        """
        Dump prompts to file in specified format
        Similar to the dump methods in prompt_generator.py
        """
        logger.info(
            f"Writing {len(prompts)} prompts to {output_file} ({format_type} format)")

        if format_type == "serve":
            self._dump_serve_format(prompts, output_file)
        elif format_type == "bench":
            self._dump_bench_format(prompts, output_file)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _dump_serve_format(self, prompts: List[Dict], output_file: str):
        """Dump in serve format (optimized batch writing)"""
        # Build all line content in batch, then write at once
        lines = []
        for prompt in prompts:
            lines.append(json.dumps({"input": prompt}) + "\n")

        # Write all content at once
        with open(output_file, "w", encoding="utf-8", buffering=8192) as outfile:
            outfile.writelines(lines)

    def _dump_bench_format(self, prompts: List[Dict], output_file: str):
        """Dump in bench format (optimized batch writing)"""
        # Build all line content in batch
        lines = []
        for idx, prompt in enumerate(prompts):
            user_content = ""
            for msg in prompt["messages"]:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break

            bench_item = {
                "task_id": idx,
                "prompt": user_content,
                # Or consider adding max_tokens parameter
                "output_tokens": prompt.get("max_tokens", 2048)
            }
            lines.append(json.dumps(
                bench_item, ensure_ascii=False, separators=(',', ':')))

        # Write all content at once, keeping consistent line breaks with prompt_generator
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write('\n'.join(lines))
            if lines:
                outfile.write('\n')

    def verify(self, prompts: List[Dict], target_tokens: int) -> Dict[str, float]:
        """Verify the generated prompts (similar to verify method in prompt_generator.py)"""
        actual_token_lengths = [prompt["num_tokens"] for prompt in prompts]
        expected_token_lengths = [prompt.get(
            "expected_tokens", target_tokens) for prompt in prompts]

        # Calculate actual token length statistics
        actual_stats = {
            "mean": float(np.mean(actual_token_lengths)),
            "std": float(np.std(actual_token_lengths)),
            "min": float(np.min(actual_token_lengths)),
            "max": float(np.max(actual_token_lengths)),
            "count": len(actual_token_lengths),
        }

        # Calculate differences from target token length
        differences = [actual - expected for actual,
                       expected in zip(actual_token_lengths, expected_token_lengths)]
        difference_stats = {
            "mean_diff": float(np.mean(differences)),
            "std_diff": float(np.std(differences)),
            "min_diff": float(np.min(differences)),
            "max_diff": float(np.max(differences)),
        }

        logger.info(f"Generated {actual_stats['count']} random prompts")
        logger.info(" Actual token length stats:")
        logger.info(
            f"  Mean: {actual_stats['mean']:.2f} ± {actual_stats['std']:.2f}")
        logger.info(
            f"  Range: {actual_stats['min']:.0f} - {actual_stats['max']:.0f}")
        logger.info(f" Token length differences (actual - expected):")
        logger.info(
            f"  Mean: {difference_stats['mean_diff']:.2f} ± {difference_stats['std_diff']:.2f}")
        logger.info(
            f"  Range: {difference_stats['min_diff']:.0f} - {difference_stats['max_diff']:.0f}")
        logger.info(f"Target length: {target_tokens}")

        return {**actual_stats, **difference_stats}


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate random prompts using random token IDs.")
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=1024,
        help="Number of tokens per prompt (default: 1024).",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Number of prompts to generate (default: 100).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1",
        help="Tokenizer name or path from HuggingFace.",
    )
    parser.add_argument(
        "--output_file_serve",
        type=str,
        default=None,
        help="Path to output file in serve format."
    )
    parser.add_argument(
        "--output_file_bench",
        type=str,
        default=None,
        help="Path to output file in bench format."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens for each request output (default: 2048).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect, max 8).",
    )
    parser.add_argument(
        "--use_parallel",
        action="store_true",
        help="Use parallel processing (default: False, use single process).",
    )
    # add random ratio
    parser.add_argument(
        "--random_ratio",
        type=float,
        default=1.0,
        help="Random ratio for input and output tokens (default: 1.0).",
    )
    parser.add_argument(
        "--custom_tokenizer",
        type=str,
        default=None,
        help="Custom tokenizer name supported by trtllm.",
        choices=["deepseek_v32", "deepseek_v4"],
    )
    parser.add_argument(
        "--vocab_from_config",
        action="store_true",
        help="Use the vocab size from the config. If not provided, will use the vocab size from the tokenizer.",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = RandomPromptGenerator(
        tokenizer_name=args.tokenizer_name, custom_tokenizer=args.custom_tokenizer,
        vocab_from_config=args.vocab_from_config)

    start_time = time.time()

    # Generate prompts
    if args.use_parallel:
        prompts = generator.generate_prompts_parallel(
            num_prompts=args.num_prompts,
            num_tokens=args.num_tokens,
            max_tokens=args.max_tokens,
            num_workers=args.num_workers,
            random_ratio=args.random_ratio
        )
    else:
        prompts = generator.generate_prompts_batch(
            num_prompts=args.num_prompts,
            num_tokens=args.num_tokens,
            max_tokens=args.max_tokens,
            random_ratio=args.random_ratio
        )

    # Write to files if specified
    if args.output_file_serve:
        generator.dump_to_file(prompts, args.output_file_serve, "serve")

    if args.output_file_bench:
        generator.dump_to_file(prompts, args.output_file_bench, "bench")

    # Verify results
    stats = generator.verify(prompts, args.num_tokens)

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logger.info(
        f"Average time per prompt: {(end_time - start_time) / args.num_prompts:.4f} seconds")


if __name__ == "__main__":
    main()
