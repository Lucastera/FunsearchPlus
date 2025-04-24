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
from implementation.multi_objective_evaluator import MultiObjectiveEvaluator

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
        # Multi-objective evaluator init
        self._use_multi_objective = False
        self._objective_weights = None
        self._multi_objective_evaluator = None
        
        self._best_per_objective = {}
        self._multi_objective_scores = []

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
    
    def enable_multi_objective_evaluation(self, enable: bool = True, weights: dict = None):
        """Enable or disable multi-objective evaluation with specified weights."""
        self._use_multi_objective = enable
        
        if enable:
            self._multi_objective_evaluator = MultiObjectiveEvaluator()
            
            self._objective_weights = weights or {
                "performance": 0.5, "simplicity": 0.2, 
                "interpretability": 0.15, "novelty": 0.15
            }
            
            self._best_per_objective = {
                "performance": {"score": float('-inf'), "function": None},
                "simplicity": {"score": float('-inf'), "function": None},
                "interpretability": {"score": float('-inf'), "function": None},
                "novelty": {"score": float('-inf'), "function": None},
                "total": {"score": float('-inf'), "function": None}
            }
        else:
            # Clear structures if disabled
            self._objective_weights = None
            self._multi_objective_evaluator = None
            self._best_per_objective = {}
            self._multi_objective_scores = []
            
    def is_valuable_solution(self, function_code: str, scores_per_test=None, threshold=0.9):
        """Check if code is valuable despite being similar to existing solutions."""
        if not self._use_multi_objective or not scores_per_test:
            return False
            
        temp_function = code_manipulation.Function(name="temp", args="", body=function_code)
        
        multi_scores = self._multi_objective_evaluator.compute_multi_objective_score(
            temp_function, scores_per_test, self._evaluated_functions, self._objective_weights
        )
        
        valuable_reasons = []
        
        for objective, data in self._best_per_objective.items():
            if objective in multi_scores["scores"] and multi_scores["scores"][objective] > data["score"] * threshold:
                valuable_reasons.append({
                    "objective": objective,
                    "current_score": multi_scores["scores"][objective],
                    "threshold": data["score"] * threshold,
                    "previous_best": data["score"]
                })
        
        if valuable_reasons:
            # 记录为什么这个解决方案被认为是有价值的
            temp_function.valuable_reasons = valuable_reasons
            print("Valuable solution reasons:")
            for reason in valuable_reasons:
                print(f"- {reason['objective']}: {reason['current_score']} > {reason['threshold']} (threshold)")
            return True
                    
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
        
        # 准备要保存的内容，添加是否是有价值的相似解决方案的标记
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score,
            'is_valuable_similar': getattr(programs, 'is_valuable_similar', False)
        }
        
        # 添加有价值解决方案的原因（如果存在）
        if hasattr(programs, 'valuable_reasons'):
            content['valuable_reasons'] = programs.valuable_reasons
        
        if hasattr(programs, 'multi_objective_scores'):
            content['multi_objective_scores'] = programs.multi_objective_scores
            
            # 添加每个目标的相对排名信息
            if self._use_multi_objective and self._multi_objective_scores:
                # 计算当前解决方案在每个目标上的排名百分比
                rankings = {}
                for objective in programs.multi_objective_scores["scores"]:
                    current_score = programs.multi_objective_scores["scores"][objective]
                    all_scores = [s["scores"][objective] for s in self._multi_objective_scores]
                    ranking = sum(1 for s in all_scores if s >= current_score) / len(all_scores)
                    rankings[objective] = ranking
                
                content['objective_rankings'] = rankings
        
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file, indent=2)

    def register_function(self, programs: code_manipulation.Function, scores_per_test=None, **kwargs):
        """Registers a function and checks for duplicates."""
        method = kwargs["method"]  # 必須從 kwargs 中獲取
        threshold = kwargs["threshold"]  # 必須從 kwargs 中獲取

        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        function_code = str(programs)
        #if self._evaluated_functions !=[] and function_code == self._evaluated_functions[0]:
        #    return
        is_duplicate = False
        if method == "hash" and self.is_duplicate_by_hash(function_code):  # 基於哈希值檢查
            is_duplicate = True
        elif method == "similarity" and self.is_duplicate_by_similarity(function_code, threshold):  # 基於相似度檢查
            is_duplicate = True
        elif method == "ai_agent" and self.is_duplicate_by_ai_agent(function_code, threshold):  # 基於 AI Agent 檢查
            is_duplicate = True
        
        # 新增标记变量，记录是否是有价值的相似解决方案
        is_valuable_similar = False
        
        if is_duplicate and self._use_multi_objective:
            if self.is_valuable_solution(function_code, scores_per_test, threshold):
                is_duplicate = False
                is_valuable_similar = True
                print("#######################################")
                print("#  Keeping valuable similar solution  #")
                print("#######################################")
                
        if is_duplicate:
            print(f"Skipping duplicate function ({method}):")
            print(function_code)
            return

        # 如果不是重複代碼，記錄哈希值和完整內容
        code_hash = hashlib.sha256(function_code.encode()).hexdigest()
        self._evaluated_hashes.add(code_hash)
        self._evaluated_functions.append(function_code)
        
        # 为函数添加是否是有价值的相似解决方案的标记
        programs.is_valuable_similar = is_valuable_similar
        
        if self._use_multi_objective and scores_per_test:
            print(f"Computing multi-objective scores with use_multi_objective={self._use_multi_objective}")
            multi_scores = self._multi_objective_evaluator.compute_multi_objective_score(
                programs, scores_per_test, self._evaluated_functions[:-1], self._objective_weights
            )
            print(f"Computed multi_scores: {multi_scores}")
            
            for objective, score in multi_scores["scores"].items():
                if score > self._best_per_objective[objective]["score"]:
                    self._best_per_objective[objective]["score"] = score
                    self._best_per_objective[objective]["function"] = programs
                    
            if multi_scores["total"] > self._best_per_objective["total"]["score"]:
                self._best_per_objective["total"]["score"] = multi_scores["total"] 
                self._best_per_objective["total"]["function"] = programs
                
            self._multi_objective_scores.append(multi_scores)
            programs.multi_objective_scores = multi_scores
            
            # 记录每一轮的分数情况
            if self._log_dir:
                self._writer.add_scalars(
                    'Multi-Objective Scores',
                    {
                        'performance': multi_scores["scores"]["performance"],
                        'simplicity': multi_scores["scores"]["simplicity"],
                        'interpretability': multi_scores["scores"]["interpretability"],
                        'novelty': multi_scores["scores"]["novelty"],
                        'total': multi_scores["total"]
                    },
                    global_step=self._num_samples
                )

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
        
        # 显示是否是有价值的相似解决方案
        if hasattr(function, 'is_valuable_similar') and function.is_valuable_similar:
            print(f'Is valuable similar solution: True')
            if hasattr(function, 'valuable_reasons'):
                print(f'Valuable reasons:')
                for reason in function.valuable_reasons:
                    print(f"  - {reason['objective']}: {reason['current_score']} > {reason['threshold']}")
        
        # 显示多目标评分（如果有）
        if hasattr(function, 'multi_objective_scores'):
            print(f'Multi-objective scores:')
            for objective, score in function.multi_objective_scores["scores"].items():
                print(f'  - {objective}: {score}')
            print(f'  - Total weighted score: {function.multi_objective_scores["total"]}')
        
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