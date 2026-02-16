import argparse
import asyncio
import aiohttp
import numpy as np
import json
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trace_build.trace import sample_reqs, bulid_workload

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, async_request_openai_completions

PROXY_PORT = None
MODEL_PATH = None
SLO_DICT = {
    "text": (0.25, 0.1),
    "image": (0.5, 0.1),
    "search": (4.0, 0.2),
    "file": (6.0, 0.2),
}

async def launch_req(idx, req_dict, is_print=True):
    global PROXY_PORT, MODEL_PATH
    req = RequestFuncInput(
        api_url = f"http://localhost:{PROXY_PORT}/v1/completions",
        model = MODEL_PATH,
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
        if not out.success:
            raise RuntimeError(f"Connection failed, please check the --model parameter.")
        result = {
            "idx": idx,
            "prompt_len": req_dict['num_tokens'],
            "output_length": req_dict['output_length'],
            "ttft": out.ttft,
            "ttft_slo": req_dict['ttft_slo'],
            "ttft_attain": req_dict['ttft_slo'] >= out.ttft,
            "tbt": sum(out.itl) / len(out.itl),
            "tbt_slo": req_dict['tbt_slo'],
            "tbt_attain": sum([i < req_dict['tbt_slo'] for i in out.itl]) / len(out.itl),
        }
        if is_print:
            print(f"req {idx}, prompt_len: {req_dict['num_tokens']}, output_length: {req_dict['output_length']}, "
            f"ttft: {out.ttft:.4f}, ttft_slo: {req_dict['ttft_slo']}, ttft attain: {result['ttft_attain']}, "
            f"tbt: {result['tbt']:.4f}s, tbt_slo: {req_dict['tbt_slo']}, tbt attain: {result['tbt_attain']:.2f}")
        return result

def res_print(results, start_time):
    num_attain = sum([item['ttft_attain'] for item in results])
    slo_attainments = num_attain / len(results)
    cost_time = time.perf_counter() - start_time
    tbt_attains = [item['tbt_attain'] for item in results]
    tbt_attain = sum(tbt_attains) / len(tbt_attains)
    print(f"cost_time: {cost_time:.4f}s")
    print(f"goodput: {num_attain / cost_time:.4f} req/s")
    print(f"TTFT slo attainment: {slo_attainments:.4f}, TBT slo attainment: {tbt_attain:.4f}")
    print(f"Avg TTFT: {sum([item['ttft'] for item in results]) / len(results):.4f}, Avg TBT: {sum([item['tbt'] for item in results]) / len(results):.4f}")
    # SLO Scaling results
    slo_scale_list = [round(i/10, 2) for i in range(4, 21, 2)]
    print(f"slo_scale_list: {slo_scale_list}")
    slo_result = []
    for slo_scale in slo_scale_list:
        num_attain = sum([item['ttft_slo']*slo_scale > item["ttft"] for item in results])
        slo_attainments = round(num_attain / len(results), 4)
        slo_result.append(slo_attainments)
    print(f"slo_attainment: {slo_result}")

async def warm_up():
    req_dict = {
        "prompt": "where is the capital of France? ", "output_length": 4, 
        "num_tokens": 7, "ttft_slo": 0.1, "tbt_slo": 0.1, "type": None}
    for i in range(32):
        out = await launch_req(i, req_dict, is_print=False)

async def benchmark(rate_scale, trace_path):
    global SLO_DICT
    sampled_reqs, trace_data = sample_reqs(trace_path, SLO_DICT)
    # Qwen Trace
    num_sampled = len(sampled_reqs)
    arrivals, intervals = bulid_workload(trace_data, num_sampled, rate_scale=1 / rate_scale)
    
    intervals = [0]+intervals.tolist()
    span = arrivals[-1]-arrivals[0]
    req_rate = num_sampled / span
    print(f"--------- Req rate: {req_rate:.2f}, rate_scale: {rate_scale}, num reqs: {len(sampled_reqs)}, span: {span:.2f}s ---------")
    
    start_time = time.perf_counter()
    tasks = []
    for idx, req_dict in enumerate(sampled_reqs):
        await asyncio.sleep(intervals[idx])
        tasks.append(asyncio.create_task(launch_req(idx, req_dict, is_print=True)))
        # tasks.append(asyncio.create_task(launch_req(idx, req_dict, is_print=False)))
    span = time.perf_counter()-start_time
    
    results = await asyncio.gather(*tasks)
    res_print(results, start_time)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--trace-path", type=str, default="trace_build/qwen_traceA_0.0min_2.0min.jsonl")
    args = parser.parse_args()

    PROXY_PORT = args.port
    MODEL_PATH = args.model

    asyncio.run(warm_up())
    asyncio.run(benchmark(args.rate_scale, args.trace_path))