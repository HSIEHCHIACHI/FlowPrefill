import argparse
import asyncio
import aiohttp
import time
import numpy as np
import matplotlib.pyplot as plt
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, async_request_openai_completions
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROXY_PORT = None
MODEL = None

async def profile_ttft(idx, req_dict, is_print=True):
    global MODEL
    req = RequestFuncInput(
        api_url = f"http://localhost:{PROXY_PORT}/v1/completions",
        model = MODEL,
        prompt = req_dict['prompt'],
        # prompt = [{"role": "user", "content": req_dict['prompt']}],
        output_len = req_dict['output_length'],
        prompt_len = req_dict['num_tokens'],
        extra_body={
            "vllm_xargs": {
                "ttft_slo": req_dict['ttft_slo'],
                "tbt_slo": req_dict['tbt_slo'],
                "arrival": time.perf_counter(),
            }
        },
    )

    async with aiohttp.ClientSession() as session:
        out = await async_request_openai_completions(req, session)
        if is_print:
            print(f"prompt_len: {req_dict['num_tokens']}, TTFT: {out.ttft:.4f}")
        return out.ttft

def get_req_dict(input_len):
    return {
        "prompt": "the " * input_len,
        "num_tokens": input_len,
        "output_length": 2,
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

async def fitting_and_plot(input_len_list, tp_size):
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
    plt.savefig(BASE_DIR / f"plotting_{MODEL.split('/')[-1]}_tp{tp_size}_online.png", bbox_inches='tight')
    plt.show()

    np.save(BASE_DIR / f"profile_{MODEL.split('/')[-1]}_tp{tp_size}_online.npy", y_fit)
    print(f"profile_{MODEL.split('/')[-1]}_tp{tp_size}_online.npy have been save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    PROXY_PORT = args.port
    MODEL = args.model

    input_len_list = [16, 32, 64, 128, 256]+[i*512 for i in range(1, 41)]
    print("input_len_list:", input_len_list)
    asyncio.run(fitting_and_plot(input_len_list, args.tp_size))
    
