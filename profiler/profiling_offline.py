import argparse
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import time
import torch

from vllm.config import KVTransferConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm import SamplingParams
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

async def gen(prompt, params, req_id):
    results_generator = asyllm.generate(prompt, params, req_id)
    final_output = None
    last_time = time.perf_counter()
    iter_times = []
    async for request_output in results_generator:
        now = time.perf_counter()
        iter_time = now - last_time
        iter_times.append(round(iter_time, 4))
        last_time = now
        final_output = request_output

    return iter_times, final_output

async def profile_ttft(idx, req_dict, is_print=True):
    prompt = req_dict["prompt"]
    ttft_slo = req_dict["ttft_slo"]
    tbt_slo = req_dict["tbt_slo"]
    arrival = time.perf_counter()
    req_id = str(idx)
    params = SamplingParams(temperature=0.0, max_tokens=1, 
                            extra_args={
                                "ttft_slo": ttft_slo,
                                "tbt_slo": tbt_slo,
                                "arrival": arrival,
                                }
                            )
    ttfts, final_output = await gen(prompt, params, req_id)
    prompt_len = len(final_output.prompt_token_ids)
    ttft = sum(ttfts) / len(ttfts)
    if is_print:
        print(f"prompt_len: {prompt_len}, ttft: {ttft:.4f}")
    return ttft

def get_req_dict(input_len):
    return {
        "prompt": "the " * input_len,
        "ttft_slo": 0.1,
        "tbt_slo": 0.1,
        "type": "profiling",
    }

async def profiling(input_len_list):
    # Warm up
    req_dict = get_req_dict(input_len=1024*2)
    for i in range(16):
        ttft = await profile_ttft(i, req_dict, is_print=False)
    # Profiling
    ttfts = []
    for input_len in input_len_list:
        req_dict = get_req_dict(input_len)
        ttft = await profile_ttft(i, req_dict, is_print=True)
        ttfts.append(ttft)
    return ttfts

async def fitting_and_plot(input_len_list, tp_size, model):
    ttfts = await profiling(input_len_list)

    def fitting(x, y):
        coef = np.polyfit(x, y, 3)
        poly = np.poly1d(coef)
        
        x_fit = np.array([i for i in range(0, max(x)+1)])
        y_fit = poly(x_fit)
        y_ = poly(x)
        return x_fit, y_fit, y_
    y = np.array(ttfts)
    x = np.array(input_len_list)
    x_fit, y_fit, y_ = fitting(x, y)
    # Clear outliers
    residuals = np.abs(y - y_)
    residuals = residuals / (residuals.max() + 1e-12)
    threshold = 0.4
    mask = residuals <= threshold
    x_clean = x[mask]
    y_clean = y[mask]
    x_fit, y_fit, _ = fitting(x_clean, y_clean)

    # plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_fit, y_fit, label='poly', color='red')
    # plt.scatter(x, y, label='原始数据', color='blue')
    plt.scatter(x_clean, y_clean, label='data', color='blue')
    plt.xlabel('Input length (token)')
    plt.ylabel('TTFT (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(BASE_DIR / f"plotting_{model.split('/')[-1]}_tp{tp_size}_offline.png", bbox_inches='tight')
    plt.show()

    np.save(BASE_DIR / f"profile_{model.split('/')[-1]}_tp{tp_size}_offline.npy", y_fit)
    print(f"profile_{model.split('/')[-1]}_tp{tp_size}_offline.npy have been save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    params = {
            "model": args.model,

            "max_model_len": 24 * 1024,
            "max_num_batched_tokens": 1024 * 8,
            "is_flowprefill": True,
            "num_runners": 64,

            "enforce_eager": True,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.8,

            "tensor_parallel_size": args.tp_size,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,

            # "kv_transfer_config": KVTransferConfig(
            #     kv_connector="NixlConnector",
            #     kv_role="kv_both",
            #     kv_port="5601",
            # ),
        }
    engine_arg = AsyncEngineArgs(**params)
    asyllm = AsyncLLMEngine.from_engine_args(engine_arg)

    input_len_list = [16, 32, 64, 128, 256]+[i*512 for i in range(1, 41)]
    print("input_len_list:", input_len_list)
    asyncio.run(fitting_and_plot(input_len_list, args.tp_size, args.model))
    
