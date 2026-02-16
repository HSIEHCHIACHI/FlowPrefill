import argparse
import json
import numpy as np
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

def load_trace(path):
    with open(path, "r") as f:
        lines = f.readlines()
        trace_data = [json.loads(line) for line in lines]

    block_size = 16
    for item in trace_data:
        if (father_id:=item['parent_chat_id']) != -1 and item['type'] == 'text':
            child = item
            father = trace_data[father_id]

            diff_hash = len(child['hash_ids'])-len(father['hash_ids'])
            child_actual_length = diff_hash * block_size
            item['input_length'] = child_actual_length
    return trace_data

def print_length_info(trace_data, lower, upper):
    qwenA_dataset = {}
    for item in trace_data:
        if item['timestamp'] < 60*lower or item['timestamp'] > 60*upper:
            continue
        arrival = []
        if item['type'] not in qwenA_dataset:
            qwenA_dataset[item['type']] = []
        req = {
            'chat_id': item['chat_id'],
            'timestamp': item['timestamp'],
            'input_length': item['input_length'],
            'output_length': item['output_length'],
        }
        qwenA_dataset[item['type']].append(req)
    inputlen_dict = {}
    outputlen_dict = {}
    for req_type, reqs in qwenA_dataset.items():
        input_lens = []
        output_lens = []
        for req in reqs:
            input_lens.append(req['input_length'])
            output_lens.append(req['output_length'])
        input_lens = np.array(input_lens)
        output_lens = np.array(output_lens)
        inputlen_dict[req_type] = input_lens
        outputlen_dict[req_type] = output_lens
        print(f"------- req_type: {req_type} num: {input_lens.size} -------")
        print(f"input length: mean: {np.mean(input_lens):.2f}, P99: {np.percentile(input_lens, 99):.2f}, Std: {np.std(input_lens):.2f}")
        print(f"output length: mean: {np.mean(output_lens):.2f}, P99: {np.percentile(output_lens, 99):.2f}, Std: {np.std(output_lens):.2f}")

def extract_datasets(dataset, lower, upper):
    new_data = []
    for req_dict in dataset:
        t = req_dict['timestamp']
        if  t >= 60*lower and t <= 60*upper:
            new_data.append(req_dict)

    with open(BASE_DIR / f"qwen_traceA_{lower}min_{upper}min.jsonl", "w", encoding="utf-8") as f:
        for r in new_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sample_reqs(trace_path, slo_dict):
    with open(trace_path, "r") as f:
        lines = f.readlines()
        trace_data = [json.loads(line) for line in lines]

    sampled_reqs = []
    input_lens = []
    input_len_dict = {}
    for item in trace_data:
        task_name = item['type']
        if task_name not in input_len_dict:
            input_len_dict[task_name] = []
        slo = slo_dict[task_name]
        req_dict = {
            "prompt": "the " * item['input_length'],
            "num_tokens": item['input_length'],
            "output_length": item['output_length'],
            "ttft_slo": slo[0],
            "tbt_slo": slo[1],
            "type": task_name,
            "arrival": item["timestamp"],
            }
        sampled_reqs.append(req_dict)
        input_lens.append(item['input_length'])
        input_len_dict[task_name].append(item['input_length'])
        
    num_sampled = len(sampled_reqs)
    span = sampled_reqs[-1]['arrival']  - sampled_reqs[0]['arrival']
    print(f"avg input len: {sum(input_lens)/len(input_lens):.2f}, max length: {max(input_lens)}, num req: {num_sampled}, span: {span/60:.2f} min")
    for req_type, input_lens in input_len_dict.items():
        input_lens = np.array(input_lens)
        print(f"------- req_type: {req_type} num: {input_lens.size} -------")
        print(f"input length: mean: {np.mean(input_lens):.2f}, P99: {np.percentile(input_lens, 99):.2f}, Std: {np.std(input_lens):.2f}")
    return sampled_reqs, trace_data

def scaling_intervals(arrivals, rate_scale):
    arr = np.sort(np.asarray(arrivals))
    dt = np.diff(arr)
    dt_new = dt * rate_scale
    arr_new = np.concatenate([[arr[0]], arr[0] + np.cumsum(dt_new)])
    return arr_new

def bulid_workload(trace_data, num_sampled, rate_scale=1.0):
    arrivals = np.array([r['timestamp'] for r in trace_data]).copy()
    arrivals = scaling_intervals(arrivals, rate_scale)
    intervals = np.diff(arrivals)
    return arrivals[:num_sampled], intervals[:num_sampled]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=BASE_DIR/"qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl")
    parser.add_argument("--lower", type=float, default=0.0, help="span lower bound(minute)")
    parser.add_argument("--upper", type=float, default=2.0, help="span upper bound(minute)")
    args = parser.parse_args()

    trace_data = load_trace(args.path)
    print_length_info(trace_data, args.lower, args.upper)
    extract_datasets(trace_data, args.lower, args.upper)


