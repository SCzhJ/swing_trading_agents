# token_controller.py
import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, Callable, Any, Type, List
from contextlib import asynccontextmanager
import logging

@dataclass
class CallRecord:
    """单次API调用的完整记录"""
    id: str
    input_tokens: int
    output_tokens: int
    timestamp: float  # 使用 time.monotonic() 避免时钟回拨
    estimate: bool  # True=预估记录, False=真实记录

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    backoff_factor: float = 1.0  # 指数退避基数: 1s, 2s, 4s, 8s...
    retry_exceptions: Tuple[Type[Exception], ...] = field(default_factory=lambda: (
        asyncio.TimeoutError,
        ConnectionError,
        Exception,  # 测试时宽松，生产环境应该更具体
    ))

@dataclass
class RequestContext:
    """
    请求上下文，用于在重试循环中传递状态
    
    使用示例：
        async with controller.acquire_slot(prompt, 100) as ctx:
            response = await api_call(ctx.call_id)
            ctx.set_result(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                result=response.choices[0].message.content
            )
    """
    call_id: str
    prompt: str
    max_output_token: int
    attempt: int = 1
    _result: Optional[Any] = None
    _input_tokens: Optional[int] = None
    _output_tokens: Optional[int] = None
    _exited: bool = False
    
    def set_result(self, *, input_tokens: int, output_tokens: int, result: Any):
        """设置API调用结果"""
        if self._exited:
            raise RuntimeError("Cannot set result after context exit!")
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._result = result
    
    @property
    def has_result(self) -> bool:
        """是否已设置结果"""
        return self._result is not None
    
    @property
    def result(self) -> Any:
        """获取结果（必须在has_result=True后调用）"""
        if not self.has_result:
            raise RuntimeError("Result not set yet")
        return self._result
    
    @property
    def input_tokens(self) -> int:
        return self._input_tokens
    
    @property
    def output_tokens(self) -> int:
        return self._output_tokens


class TokenController:
    """
    使用方式（推荐）：
        async with controller.acquire_slot("prompt", 100) as ctx:
            response = await api_call(ctx.call_id)
            ctx.set_result(...)
            return ctx.result
    """
    
    def __init__(
        self,
        tpm: int,
        rpm: int,
        max_concurrent: int,
        provider: str,
        token_estimator: Optional[Callable[[str], int]] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.tpm = tpm
        self.rpm = rpm
        self.max_concurrent = max_concurrent
        self.provider = provider

        # 核心数据状态
        self.records: Deque[CallRecord] = deque()
        self.in_flight: Dict[str, asyncio.Task] = {}
        
        # 异步同步原语
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 性能优化计数器
        self._tpm_counter = 0
        self._rpm_counter = 0

        # Token估算器
        self.token_estimator = token_estimator or self._default_estimator
        
        # 重试配置
        self.retry_config = retry_config or RetryConfig()
        
        self.logger = logging.getLogger(provider)

        
        self.logger.info(
            f"TokenController initialized: TPM={tpm}, RPM={rpm}, "
            f"Concurrency={max_concurrent}, Provider={provider}, "
            f"MaxRetry={self.retry_config.max_attempts}"
        )

    def _default_estimator(self, prompt: str) -> int:
        """默认Token估算器（保守策略）"""
        return max(1, len(prompt) // 4 + 10)

    async def token_estimate(self, prompt: str) -> int:
        """异步估算输入token数"""
        return self.token_estimator(prompt)

    async def _cleanup_expired(self) -> None:
        """清理60秒前的过期记录"""
        now = time.monotonic()
        cutoff_time = now - 60.0
        
        async with self.lock:
            expired_records = 0
            skipped_est_count = 0
            
            temp_est_deque = deque()

            while self.records and self.records[0].timestamp < cutoff_time:
                record = self.records.popleft()
                if record.estimate:
                    temp_est_deque.append(record)
                    skipped_est_count += 1
                else:
                    self._tpm_counter -= (record.input_tokens + record.output_tokens)
                    self._rpm_counter -= 1
                    expired_records += 1

            if temp_est_deque:
                self.records.extendleft(reversed(temp_est_deque)) 
            if expired_records > 0 or skipped_est_count > 0:
                self.logger.info(
                    f"Expiring cleanup: removed {expired_records} real records, "
                    f"skipped {skipped_est_count} estimate records"
                )

    def _get_current_load(self) -> Tuple[int, int, int]:
        """获取当前负载（O(1)）"""
        # 必须在锁内调用
        return self._tpm_counter, self._rpm_counter, len(self.in_flight)

    async def _wait_for_capacity(
        self, 
        required_tokens: int, 
        call_id: str
    ) -> None:
        """等待直到有足够容量"""
        wait_start = time.monotonic()
        total_wait_time = 0.0
        
        while True:
            await self._cleanup_expired()
            
            async with self.lock:
                current_tpm, current_rpm, current_concurrent = self._get_current_load()
                
                tpm_ok = current_tpm + required_tokens <= self.tpm
                rpm_ok = current_rpm < self.rpm
                
                if tpm_ok and rpm_ok:
                    if total_wait_time > 0:
                        self.logger.debug(
                            f"Request {call_id} waited {total_wait_time:.3f}s: "
                            f"TPM={current_tpm}/{self.tpm}, RPM={current_rpm}/{self.rpm}"
                        )
                    return
                
                reasons = []
                if not tpm_ok: 
                    reasons.append(f"TPM({current_tpm}+{required_tokens}>{self.tpm})")
                if not rpm_ok: 
                    reasons.append(f"RPM({current_rpm}>={self.rpm})")
                self.logger.debug(f"Request {call_id} waiting: {', '.join(reasons)}")
            
            await asyncio.sleep(0.05)
            total_wait_time = time.monotonic() - wait_start

    async def wait_before_call_if_needed(
        self,
        prompt: str,
        max_output_token: int
    ) -> str:
        """底层API：调用前准备（不自动重试）"""
        call_id = str(uuid.uuid4())
        input_est = await self.token_estimate(prompt)
        required_tokens = input_est + max_output_token
        
        self.logger.debug(
            f"Preparing {call_id}: est_input={input_est}, "
            f"max_output={max_output_token}, required={required_tokens}"
        )
        
        # 问一下ai以下可不可行
        await self._wait_for_capacity(required_tokens, call_id)
        await self.semaphore.acquire()
        
        record = CallRecord(
            id=call_id,
            input_tokens=input_est,
            output_tokens=max_output_token,
            timestamp=time.monotonic(),
            estimate=True
        )
        
        async with self.lock:
            self.records.append(record)
            self._tpm_counter += (input_est + max_output_token)
            self._rpm_counter += 1
            self.in_flight[call_id] = asyncio.current_task()
        
        self.logger.debug(f"Request {call_id} approved and locked")
        return call_id

    async def wait_after_call_if_needed(
        self,
        call_id: str,
        actual_input_tokens: int,
        actual_output_tokens: int
    ) -> None:
        """底层API：调用后更新（不自动重试）"""
        real_record = CallRecord(
            id=call_id,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            timestamp=time.monotonic(),
            estimate=False
        )
        
        async with self.lock:
            found = False
            for i, rec in enumerate(self.records):
                if rec.id == call_id and rec.estimate:
                    old_tokens = rec.input_tokens + rec.output_tokens
                    new_tokens = actual_input_tokens + actual_output_tokens
                    token_delta = new_tokens - old_tokens
                    
                    self.records[i] = real_record
                    found = True
                    self._tpm_counter += token_delta
                    
                    self.logger.debug(
                        f"Replaced {call_id}: {old_tokens} -> {new_tokens} "
                        f"({'+' if token_delta>=0 else ''}{token_delta})"
                    )
                    break
            
            if not found:
                raise ValueError(f"Record {call_id} not found for replacement")
                # Logic without raising exception:
                # self.logger.warning(f"Record {call_id} not found for replacement")
                # self.records.append(real_record)
                # self._tpm_counter += (actual_input_tokens + actual_output_tokens)
            
            self.in_flight.pop(call_id, None)
        
        self.semaphore.release()

    async def cleanup_call(self, call_id: str) -> None:
        """底层API：异常清理"""
        async with self.lock:
            matching = [r for r in self.records if r.id == call_id]
            
            self.records = deque([r for r in self.records if r.id != call_id])
            
            for rec in matching:
                self._tpm_counter -= (rec.input_tokens + rec.output_tokens)
                self._rpm_counter -= 1
            
            was_in_flight = self.in_flight.pop(call_id, None) is not None
        
        if was_in_flight:
            self.semaphore.release()
            self.logger.info(f"Cleaned {call_id}: removed {len(matching)} records")
        else:
            self.logger.debug(f"Cleanup {call_id}: nothing to do")

    # ======== 新增：集成重试机制 ========

    @asynccontextmanager
    async def acquire_slot(
        self,
        prompt: str | List[Dict[str, str]],
        max_output_token: int,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        高级API：带自动重试的请求槽位上下文管理器
        
        使用示例：
            async with controller.acquire_slot("prompt", 100) as ctx:
                response = await client.chat.completions.create(...)
                ctx.set_result(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    result=response.choices[0].message.content
                )
                return ctx.result
        
        异常处理：
            - 如果API调用失败，会自动重试
            - 如果重试耗尽，会清理资源并抛出最后一次异常
            - 如果用户在with块内抛出异常，也会自动清理
        """
        config = retry_config or self.retry_config
        prompt = str(prompt)
        
        # 重试循环
        last_exception = None
        call_id = None
        
        for attempt in range(1, config.max_attempts + 1):
            ctx = RequestContext(
                call_id="PENDING",  # 临时值，下面会更新
                prompt=prompt,
                max_output_token=max_output_token,
                attempt=attempt
            )
            
            try:
                # 阶段1：获取配额
                call_id = await self.wait_before_call_if_needed(prompt, max_output_token)
                ctx.call_id = call_id  # 更新真实的call_id
                
                self.logger.info(
                    f"🔄 Attempt {attempt}/{config.max_attempts} for '{prompt[:30]}...', "
                    f"call_id={call_id}"
                )
                
                # 阶段2：执行用户代码（在with块内）
                yield ctx
                
                if not ctx.has_result:
                    # 用户没有调用set_result，这是编程错误
                    raise RuntimeError(
                        f"Request {call_id}: ctx.set_result() was never called! "
                        f"You must call ctx.set_result(input_tokens=..., output_tokens=..., result=...) "
                        f"within the 'async with' block."
                        )

                await self.wait_after_call_if_needed(
                    call_id,
                    ctx.input_tokens,
                    ctx.output_tokens
                )
                self.logger.info(f"✅ Request {call_id} succeeded on attempt {attempt}")

                ctx._exited = True
                # 成功，退出重试循环
                break
                
            except config.retry_exceptions as e:
                last_exception = e
                
                # 确保资源被清理
                if call_id:
                    self.logger.debug(f"Cleaning up {call_id} after failure on attempt {attempt}")
                    await self.cleanup_call(call_id)
                    call_id = None
                
                # 如果不是最后一次，等待后重试
                if attempt < config.max_attempts:
                    wait_time = config.backoff_factor * (2 ** (attempt - 1))
                    self.logger.warning(
                        f"❌ Attempt {attempt} failed: {type(e).__name__}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"💥 All {config.max_attempts} attempts failed for '{prompt[:30]}...'"
                    )
                    ctx._exited = True
                    raise  # 重试耗尽，抛出异常
                    
            except Exception:
                # 非预期异常，立即清理并抛出
                if call_id:
                    await self.cleanup_call(call_id)
                ctx._exited = True
                raise
        
        else:
            # 理论上不会到达这里，但为类型检查加保险
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Unexpected retry logic error")
