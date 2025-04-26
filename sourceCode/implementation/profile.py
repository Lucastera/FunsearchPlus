# implemented by RZ
# profile the experiment using tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from implementation import code_manipulation
from torch.utils.tensorboard import SummaryWriter
import hashlib  # 新增導入
from difflib import SequenceMatcher  # 用於計算相似度
from openai import OpenAI

import http.client

class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        # self._pkl_dir = pkl_dir
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        self._evaluated_hashes = set()  # 存儲已評估代碼的哈希值
        self._evaluated_functions = []  # 存儲已評估代碼的完整內容
        self._load_evaluated_hashes()

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _load_evaluated_hashes(self):
        """從日志文件夾加載已評估代碼的哈希值和完整內容。"""
        if not self._json_dir:
            return
        for file_name in os.listdir(self._json_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(self._json_dir, file_name), 'r') as json_file:
                    content = json.load(json_file)
                    function_code = content.get('function', '')
                    code_hash = hashlib.sha256(function_code.encode()).hexdigest()
                    self._evaluated_hashes.add(code_hash)
                    self._evaluated_functions.append(function_code)

    def is_duplicate_by_hash(self, function_code: str) -> bool:
        """檢查代碼是否已評估過（基於哈希值）。"""
        code_hash = hashlib.sha256(function_code.encode()).hexdigest()
        return code_hash in self._evaluated_hashes

    def is_duplicate_by_similarity(self, function_code: str, threshold: float = 0.9) -> bool:
        """檢查代碼是否已評估過（基於相似度）。"""
        for evaluated_code in self._evaluated_functions:
            similarity = SequenceMatcher(None, function_code, evaluated_code).ratio()
            if similarity >= threshold:
                return True
        return False
    
    def is_duplicate_by_ai_agent(self, function_code: str, threshold: float = 0.9) -> bool:
        """
        檢查代碼是否已評估過（基於 AI Agent 評估）。
        Args:
            function_code: 要檢查的代碼。
            threshold: 相似性分數的閾值，默認為 0.9。
        Returns:
            如果代碼被認為是重複的，返回 True；否則返回 False。
        """
        for evaluated_code in self._evaluated_functions:
            try:
                # 使用 AI Agent 比較代碼相似性
                conn = http.client.HTTPSConnection("api.deepseek.com")
                payload = json.dumps({
                    "max_tokens": 512,
                    "model": "deepseek-chat",
                    "messages": [
                                {"role": "system", "content": "You are a code similarity analyzer. Compare the two code snippets and return only a similarity score from [0, 1] based on their aim and logic. 1 means identical, 0 means completely different. Output should be a single number."},
                                {"role": "user", "content": f"Code 1:\n{function_code}\n\nCode 2:\n{evaluated_code}"}]
                })
                headers = {
                    'Authorization': 'Bearer sk-4d4b1fb4def14ae3887a21683c3f1763',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)

                words_to_remove = ['similarity','score', 'Score', 'Similarity',':',  ' ']
                score_text = data['choices'][0]['message']['content'].strip()
                print(f"AI Agent 評估結果: {score_text}")

                for word in words_to_remove:
                    score_text = score_text.replace(word, '')
                score = float(score_text)
                if score >= threshold:
                    return True
            except Exception as e:
                logging.error(f"AI Agent 評估失敗: {e}")
                continue
        return False

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )

    def _write_json(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function, **kwargs):
        
        """Registers a function and checks for duplicates.
        method = kwargs["method"]  # 必須從 kwargs 中獲取
        threshold = kwargs["threshold"]  # 必須從 kwargs 中獲取

        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return
        
        function_code = str(programs)
        if self._evaluated_functions !=[] and function_code == self._evaluated_functions[0]:
            return
        if method == "hash" and self.is_duplicate_by_hash(function_code):  # 基於哈希值檢查
            print("#########################################")
            print("#  Skipping duplicate function (hash):  #")
            print("#########################################")
            print(function_code)
            return
        elif method == "similarity" and self.is_duplicate_by_similarity(function_code, threshold):  # 基於相似度檢查
            print("###############################################")
            print("#  Skipping duplicate function (similarity):  #")
            print("###############################################")
            print(function_code)
            return
        elif method == "ai_agent" and self.is_duplicate_by_ai_agent(function_code, threshold):  # 基於 AI Agent 檢查
            print("###############################################")
            print("#  Skipping duplicate function (AI Agent):    #")
            print("###############################################")
            print(function_code)
            return


        # 如果不是重複代碼，記錄哈希值和完整內容
        code_hash = hashlib.sha256(function_code.encode()).hexdigest()
        self._evaluated_hashes.add(code_hash)
        self._evaluated_functions.append(function_code)
        """
        
        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        # function_name = function.name
        # function_body = function.body.strip('\n')
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

        # update ...
        # self._each_sample_best_program_score.append(self._cur_best_program_score)
        # self._each_sample_evaluate_success_program_num.append(self._evaluate_success_program_num)
        # self._each_sample_evaluate_failed_program_num.append(self._evaluate_failed_program_num)
        # self._each_sample_tot_sample_time.append(self._tot_sample_time)
        # self._each_sample_tot_evaluate_time.append(self._tot_evaluate_time)
