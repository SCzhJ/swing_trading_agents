# token_tracker.py
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

class TokenTracker:
    def __init__(self, warning_price_threshold: float=15.0):
        self.project_root = self._find_project_root()
        self.tracker_file = self.project_root / "logs/token_usage/estimated_deposit.json"
        self.warning_price_threshold = warning_price_threshold

        # 加载配置文件
        with open(self.project_root / "configs/provider_infos.json", "r", encoding="utf-8") as f:
            self.price_sheet = json.load(f)

        self.usage_records: List[Dict[str, Any]] = []   # 每次调用
        self.logger = logging.getLogger(__name__)

    def _find_project_root(self) -> Path:
        """自动找 pyproject.toml 所在目录"""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
        raise FileNotFoundError("pyproject.toml 未找到，无法确定项目根目录")
    
    def update_deposit(self):
        # list all file under logs/token_usage/deposit_update/
        with open(self.tracker_file, "r", encoding="utf-8") as f_deposit:
            deposit_data = json.load(f_deposit)
            deposit_update_dir = self.project_root / "logs/token_usage/deposit_update/"
            for file in deposit_update_dir.iterdir():
                if file.suffix == ".json":
                    provider = file.stem
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # check whether the deposit_data has the provider, if not then add to it
                        if provider not in deposit_data["last_updated_deposit"]:
                            deposit_data["last_updated_deposit"][provider] = {
                                "time": data["time"],
                                "deposit": data["deposit"],
                                "recorded": True
                            }
                            self.logger.info(f"新增 {provider} 的初始估计存款: {data['deposit']}")
                        else:
                            if datetime.fromisoformat(data["time"]) > datetime.fromisoformat(deposit_data["last_updated_deposit"][provider]["time"]):
                                deposit_data["last_updated_deposit"][provider] = {
                                    "time": data["time"],
                                    "deposit": data["deposit"],
                                    "recorded": True
                                }
                                self.logger.info(f"更新 {provider} 的估计存款为: {data['deposit']}")
                            else:
                                self.logger.info(f"{provider} 的估计存款未更新, \
                                                当前记录为: 时间: {deposit_data['last_updated_deposit'][provider]["time"]}, 存款: {deposit_data['last_updated_deposit'][provider]["deposit"]}")
        with open(self.tracker_file, "w", encoding="utf-8") as f_deposit:
            json.dump(deposit_data, f_deposit, ensure_ascii=False, indent=4)

    def track_usage(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """记录一次 LLM 调用（只存内存）"""
        if provider not in self.price_sheet or model not in self.price_sheet[provider]:
            raise ValueError(f"未知模型: {provider}/{model}")

        self.usage_records.append({
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat(),
        })

    def finalize(self):
        """程序结束：计算费用 + 写文件 + 警报"""
        if not self.usage_records:
            return  # 没调用，直接返回

        lines = []
        total_cost: Dict[str, float] = {}

        # 1. 遍历每条记录，计算费用
        for rec in self.usage_records:
            provider = rec["provider"]
            model = rec["model"]
            pricing = self.price_sheet[provider][model]
            # if rec[provider][model] has fixed_price
            if "fixed_price" in pricing:
                cost = pricing["fixed_price"]
            elif "input_token_price" in pricing and "output_token_price" in pricing:
                cost = (
                    rec["input_tokens"] * pricing["input_token_price"] +
                    rec["output_tokens"] * pricing["output_token_price"]
                )
            else:
                raise ValueError(f"模型 {provider}/{model}, 未知计费标准")

            total_cost[provider] = total_cost.get(provider, 0.0) + cost

            lines.append(json.dumps({
                "event": "usage",
                "timestamp": rec["timestamp"],
                "provider": provider,
                "model": model,
                "input_tokens": rec["input_tokens"],
                "output_tokens": rec["output_tokens"],
                "cost_usd": round(cost, 10),
            }, ensure_ascii=False))

        self.logger.info(lines)
        # 2. 写文件（一次性！）
        save_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_token_usage.json"
        save_file_path = self.project_root / "logs/token_usage/detailed_usage/" / save_file_name
        
        with open(save_file_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        self.logger.info(f"Token 记录已保存到: {save_file_path}")
        
        with open(self.tracker_file, "r", encoding="utf-8") as f:
            deposit_data = json.load(f)
        for provider, cost in total_cost.items():
            deposit_data["last_updated_deposit"][provider]["deposit"] -= cost
            if deposit_data["last_updated_deposit"][provider]["deposit"] < self.warning_price_threshold:
                self.logger.warning(f"{provider} 的估计存款已低于阈值 {self.warning_price_threshold} USD")
        with open(self.tracker_file, "w", encoding="utf-8") as f:
            json.dump(deposit_data, f, ensure_ascii=False, indent=4)

    def get_summary(self) -> Dict[str, float]:
        """返回当前内存中的总结"""
        if not self.usage_records:
            return {}  # 没调用，返回空字典

        total_cost: Dict[str, float] = {}

        for rec in self.usage_records:
            provider = rec["provider"]
            model = rec["model"]
            pricing = self.price_sheet[provider][model]
            # 处理不同的定价方式
            if "fixed_price" in pricing:
                cost = pricing["fixed_price"]
            elif "input_token_price" in pricing and "output_token_price" in pricing:
                cost = (
                    rec["input_tokens"] * pricing["input_token_price"] +
                    rec["output_tokens"] * pricing["output_token_price"]
                )
            else:
                raise ValueError(f"模型 {provider}/{model}, 未知计费标准")
            
            total_cost[provider] = total_cost.get(provider, 0.0) + cost
        return total_cost