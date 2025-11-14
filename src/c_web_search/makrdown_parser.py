# ======================================
# MarkdownParser - 声明式Markdown解析引擎
# ======================================

from typing import List, Tuple, Union
from abc import ABC
from dataclasses import dataclass

# ============================================================================
# 一、模式元素定义 - 类型安全的声明式结构
# ============================================================================

@dataclass
class RegexAlternatives:
    """
    正则表达式候选项：匹配其中任意一个正则模式
    
    示例：
        RegexAlternatives(["##\\s+Start", "#\\s+Begin"])  # 匹配二级或一级标题
    """
    patterns: List[str]
    
    def __post_init__(self):
        """验证：至少包含一个模式"""
        if not self.patterns:
            raise InvalidPatternError("RegexAlternatives must have at least one pattern")
    
    def __repr__(self):
        return f"RegexAlternatives({self.patterns})"


@dataclass
class SkipExact:
    """
    精确跳过：跳过指定数量的行
    
    示例：
        SkipExact(2)  # 精确跳过2行
    """
    count: int
    
    def __post_init__(self):
        """验证：跳过行数必须为非负整数"""
        if not isinstance(self.count, int) or self.count < 0:
            raise InvalidPatternError(f"SkipExact count must be non-negative integer, got {self.count}")
    
    def __repr__(self):
        return f"SkipExact({self.count})"


@dataclass
class SkipRange:
    """
    范围跳过：跳过一定数量的行（在min和max之间）
    
    示例：
        SkipRange(0, 5)  # 跳过0到5行（灵活匹配）
        SkipRange(3, 3)  # 等价于 SkipExact(3)
    
    注意：
        - 当处于PatternSequence的开头时，策略决定跳过多少行
        - 当处于PatternSequence的中间时，通过上下文确定具体跳过多少行
    """
    min_lines: int
    max_lines: int
    
    def __post_init__(self):
        """验证：范围必须有效且非负"""
        if not isinstance(self.min_lines, int) or self.min_lines < 0:
            raise InvalidPatternError(f"min_lines must be non-negative integer, got {self.min_lines}")
        if not isinstance(self.max_lines, int) or self.max_lines < 0:
            raise InvalidPatternError(f"max_lines must be non-negative integer, got {self.max_lines}")
        if self.min_lines > self.max_lines:
            raise InvalidPatternError(f"min_lines ({self.min_lines}) cannot exceed max_lines ({self.max_lines})")
    
    def __repr__(self):
        return f"SkipRange({self.min_lines}, {self.max_lines})"


# 类型别名：模式序列（一个或多个模式元素的组合）
PatternElement = Union[RegexAlternatives, SkipExact, SkipRange]
PatternSequence = List[PatternElement]


# ============================================================================
# 二、异常体系 - 精确的错误诊断与定位
# ============================================================================


class MarkdownParserError(Exception):
    """所有MarkdownParser异常的基类"""
    pass


class InvalidPatternError(MarkdownParserError):
    """
    模式定义无效错误
    
    场景：
        - RegexAlternatives传入空列表
        - SkipRange的min > max
        - SkipExact传入负数
    """
    pass


class CursorOutOfBoundsError(MarkdownParserError):
    """
    起始位置越界错误
    
    场景：
        - start_from参数大于Markdown总行数
        - parser.cursor被错误地设置到文本末尾之后
    """
    
    def __init__(self, cursor: int, total_lines: int):
        self.cursor = cursor
        self.total_lines = total_lines
    
    def __str__(self):
        return f"Cursor {self.cursor} is out of bounds (total lines: {self.total_lines})"


class PatternNotFoundError(MarkdownParserError):
    """
    模式未能匹配错误
    
    场景：
        - 在指定位置找不到任何匹配的模式
        - 匹配到中途失败，无法完成整个序列
        
    属性：
        pattern: 失败的PatternSequence
        start_line: 起始行号
        context: 匹配失败的上下文文本（最多5行）
    """
    
    def __init__(self, pattern: PatternSequence, start_line: int, context: List[str]):
        self.pattern = pattern
        self.start_line = start_line
        self.context = context
    
    def __str__(self):
        preview = "\\n  ".join(self.context[:5])  # 只显示前5行
        return (f"Failed to match pattern {self.pattern}\\n"
                f"Starting from line {self.start_line}\\n"
                f"Context:\\n  {preview}")


class AmbiguousPatternError(MarkdownParserError):
    """
    模式定义模糊错误
    
    场景：
        - SkipRange在PatternSequence开头，且strategy无法确定跳过多少行
        - 多个可能的匹配路径，无法确定唯一解
    """
    pass


# ============================================================================
# 三、MarkdownParser 核心实现
# ============================================================================


class MarkdownParser:
    """
    声明式Markdown解析引擎
    
    核心功能：
        1. extract_between: 提取两个垂直标记之间的内容块
        2. match_sequence: 匹配三段连续模式并返回目标段
    
    设计哲学：
        - 类型安全：所有模式必须通过工厂方法创建
        - 自文档化：代码即意图表达
        - 精确错误：提供详细的匹配失败诊断信息
        - 状态管理：内置cursor支持迭代式解析
    
    示例：
        # 创建解析器
        parser = MarkdownParser("# Title\\n\\n## Section 1\\nContent 1\\n\\n## Section 2\\nContent 2")
        
        # 提取Section 1和Section 2之间的内容
        end_cursor, content = parser.extract_between(
            start_marker=[parser.any_of("## Section 1")],
            end_marker=[parser.any_of("## Section 2")],
            inclusive=False
        )
        # content: ["Content 1"]
    """
    
    def __init__(self, markdown: str, strategy: str = "lazy"):
        """
        初始化解析器
        
        Args:
            markdown: 要解析的Markdown文本字符串
            strategy: SkipRange匹配策略
                - "lazy": 跳过最少行数（优先取min_lines）
                - "greedy": 跳过最多行数（优先取max_lines）
                - "first": 第一个有效的匹配
        
        抛出:
            TypeError: markdown不是字符串
            ValueError: strategy不是支持的策略之一
        """
        if not isinstance(markdown, str):
            raise TypeError(f"markdown must be a string, got {type(markdown).__name__}")
        
        if strategy not in ("lazy", "greedy", "first"):
            raise ValueError(f"strategy must be 'lazy', 'greedy' or 'first', got '{strategy}'")
        
        self.original_text = markdown
        # 按行分割，保留换行符以便重构原始文本
        self.lines = markdown.splitlines(keepends=True)
        self.cursor = 0  # 当前解析位置（行号）
        self.strategy = strategy
    
    # ------------------------------------------------------------------------
    # 四、工厂方法 - 创建类型安全的模式元素
    # ------------------------------------------------------------------------
    
    @staticmethod
    def any_of(*patterns: str) -> RegexAlternatives:
        """
        创建正则表达式候选项（匹配其中任意一个）
        
        Args:
            *patterns: 一个或多个正则表达式字符串
        
        返回:
            RegexAlternatives实例
        
        示例:
            parser.any_of("##\\s+Start", "#\\s+Begin", "START")
        """
        return RegexAlternatives(list(patterns))
    
    @staticmethod
    def skip(count: int) -> SkipExact:
        """
        创建精确跳过模式（跳过指定数量的行）
        
        Args:
            count: 要跳过的行数（必须为非负整数）
        
        返回:
            SkipExact实例
        
        示例:
            parser.skip(2)  # 跳过2行
        
        抛出:
            InvalidPatternError: count为负数
        """
        return SkipExact(count)
    
    @staticmethod
    def skip_between(min_lines: int, max_lines: int) -> SkipRange:
        """
        创建范围跳过模式（跳过min到max行之间的数量）
        
        Args:
            min_lines: 最少跳过的行数
            max_lines: 最多跳过的行数
        
        返回:
            SkipRange实例
        
        示例:
            parser.skip_between(0, 5)  # 跳过0到5行
            parser.skip_between(3, 3)  # 等价于 skip(3)
        
        抛出:
            InvalidPatternError: min_lines > max_lines 或参数为负
        """
        return SkipRange(min_lines, max_lines)
    
    # ------------------------------------------------------------------------
    # 五、内部验证与匹配逻辑
    # ------------------------------------------------------------------------
    
    def _validate_cursor(self, line_idx: int):
        """验证行号是否在有效范围内"""
        if not (0 <= line_idx <= len(self.lines)):
            raise CursorOutOfBoundsError(line_idx, len(self.lines))
    
    def _match_regex_alternatives(self, line: str, element: RegexAlternatives) -> bool:
        """
        尝试用正则候选项匹配一行
        
        返回:
            True: 任一正则匹配成功
            False: 全部匹配失败
        """
        for pattern in element.patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _match_at_position(
        self,
        start_line: int,
        pattern: PatternSequence
    ) -> Tuple[int, List[int]]:
        """
        在指定位置尝试匹配整个模式序列
        
        这是核心算法，负责处理各种类型的模式元素
        
        Args:
            start_line: 开始匹配的行号
            pattern: 要匹配的模式序列
        
        返回:
            Tuple[matched_until, matched_line_indices]
            - matched_until: 匹配到的最后一行的下一行行号
            - matched_line_indices: 实际匹配到的所有行号列表（调试用）
        
        抛出:
            PatternNotFoundError: 匹配失败
        """
        current_line = start_line
        matched_lines = []  # 记录匹配到的行号
        
        for i, element in enumerate(pattern):
            if isinstance(element, RegexAlternatives):
                # 在指定位置尝试任一正则
                if current_line >= len(self.lines):
                    raise PatternNotFoundError(pattern, start_line, self.lines[start_line:start_line+3])
                
                if self._match_regex_alternatives(self.lines[current_line].strip(), element):
                    matched_lines.append(current_line)
                    current_line += 1
                else:
                    # 匹配失败，提供上下文
                    context = self.lines[max(0, current_line-2):current_line+3]
                    raise PatternNotFoundError(pattern, start_line, context)
            
            elif isinstance(element, SkipExact):
                # 精确跳过：直接推进cursor
                if current_line + element.count > len(self.lines):
                    raise PatternNotFoundError(pattern, start_line, self.lines[current_line:current_line+3])
                # SkipExact不记录matched_lines，因为它是"跳过"不是"匹配"
                current_line += element.count
            
            elif isinstance(element, SkipRange):
                # 范围跳过：最复杂的逻辑
                remaining_lines = len(self.lines) - current_line
                
                # 确定跳过多少行
                if i == 0:  # 在模式开头
                    # 需要反向匹配后续元素来确定跳过多少行
                    if len(pattern) == 1:
                        raise AmbiguousPatternError("SkipRange cannot be the only element in pattern")
                    
                    # 尝试从min到max的每一个可能值
                    for skip_count in range(element.min_lines, element.max_lines + 1):
                        test_cursor = current_line + skip_count
                        if test_cursor >= len(self.lines):
                            break
                        
                        # 尝试匹配后续元素
                        try:
                            self._match_at_position(test_cursor, pattern[i+1:])
                            # 成功后缀匹配，说明这个skip_count可行
                            current_line = test_cursor
                            break
                        except PatternNotFoundError:
                            continue  # 尝试下一个skip_count
                    else:
                        # 所有可能性都失败
                        raise PatternNotFoundError(pattern, start_line, self.lines[current_line:current_line+3])
                
                elif i == len(pattern) - 1:  # 在模式末尾
                    # 策略决定跳过多少行
                    if self.strategy == "lazy":
                        skip_count = element.min_lines
                    elif self.strategy == "greedy":
                        skip_count = min(element.max_lines, remaining_lines)
                    elif self.strategy == "first":
                        skip_count = element.min_lines
                    else:
                        skip_count = element.min_lines
                    
                    current_line += skip_count
                
                else:  # 在模式中间
                    # 通过匹配前后元素来确定跳过多少行
                    # 这是双向匹配，计算量较大
                    best_skip = None
                    for skip_count in range(element.min_lines, element.max_lines + 1):
                        test_cursor = current_line + skip_count
                        if test_cursor >= len(self.lines):
                            break
                        
                        try:
                            # 尝试匹配后续元素
                            self._match_at_position(test_cursor, pattern[i+1:])
                            best_skip = skip_count
                            if self.strategy == "first":
                                break  # 找到第一个可行解
                        except PatternNotFoundError:
                            continue
                    
                    if best_skip is None:
                        raise PatternNotFoundError(pattern, start_line, self.lines[current_line:current_line+3])
                    
                    current_line += best_skip
        
        return current_line, matched_lines
    
    # ------------------------------------------------------------------------
    # 六、核心公共API
    # ------------------------------------------------------------------------
    
    def extract_between(
        self,
        start_marker: PatternSequence,
        end_marker: PatternSequence,
        inclusive: bool = False,
        must_exist: bool = True
    ) -> Tuple[int, List[str]]:
        """
        提取两个垂直标记之间的内容块
        
        这是主要的提取方法，用于"剪切"Markdown的垂直区间。
        它会从当前cursor位置开始搜索start_marker，然后提取直到end_marker的内容。
        
        Args:
            start_marker: 起始边界标记（above_pattern）
            end_marker: 结束边界标记（below_pattern）
            inclusive: 是否将标记行包含在返回结果中
            must_exist: 是否必须找到标记（False则返回空列表而不是抛异常）
        
        返回:
            Tuple[end_cursor, content_lines]
            - end_cursor: 匹配结束后的新cursor位置
            - content_lines: 提取的内容行列表（按行分割）
        
        抛出:
            PatternNotFoundError: 如果must_exist=True但标记未找到
            InvalidPatternError: 如果模式序列定义无效
        
        示例:
            # 提取两个标题之间的内容（不包含标题）
            parser.extract_between(
                start_marker=[parser.any_of("## Start")],
                end_marker=[parser.any_of("## End")],
                inclusive=False
            )
        """
        # 验证模式序列有效性
        if not start_marker or not end_marker:
            raise InvalidPatternError("start_marker and end_marker cannot be empty")
        
        try:
            # 从当前cursor开始搜索start_marker
            start_line = self.cursor
            while start_line < len(self.lines):
                try:
                    # 尝试在当前位置匹配start_marker
                    start_end, _ = self._match_at_position(start_line, start_marker)
                    break  # 匹配成功
                except PatternNotFoundError:
                    start_line += 1  # 移动到下一行继续搜索
            else:
                # 搜索到文件末尾仍未找到
                if must_exist:
                    raise PatternNotFoundError(start_marker, self.cursor, self.lines[-5:])
                return self.cursor, []
            
            # 匹配start_marker成功，现在搜索end_marker
            # 如果inclusive=False，从start_marker结束位置的下一行开始
            search_start = start_end if not inclusive else start_line
            
            end_line = search_start
            while end_line < len(self.lines):
                try:
                    # 尝试在当前位置匹配end_marker
                    end_cursor, _ = self._match_at_position(end_line, end_marker)
                    break  # 匹配成功
                except PatternNotFoundError:
                    end_line += 1  # 移动到下一行继续搜索
            else:
                # 搜索到文件末尾仍未找到end_marker
                if must_exist:
                    raise PatternNotFoundError(end_marker, search_start, self.lines[search_start:search_start+5])
                # 如果不要求必须存在，返回到末尾的内容
                return len(self.lines), self.lines[search_start:]
            
            # 成功找到start_marker和end_marker
            if inclusive:
                # 包含标记行，内容从start_line到end_cursor
                content = self.lines[start_end:end_cursor]
            else:
                # 不包含标记行，内容从start_marker之后到end_marker
                content = self.lines[search_start:end_line]
            
            return end_cursor, content
        
        except Exception as e:
            if must_exist:
                raise e
            return self.cursor, []
    
    def match_sequence(
        self,
        target: PatternSequence,
        prefix: Optional[PatternSequence] = None,
        suffix: Optional[PatternSequence] = None,
        start_from: Optional[int] = None,
        allow_partial: bool = False
    ) -> Tuple[int, List[str]]:
        """
        匹配三段连续模式并返回目标段
        
        这是主要的模式匹配方法，用于精确提取符合特定结构的内容。
        它确保prefix（如果提供）、target和suffix（如果提供）按顺序连续出现。
        
        Args:
            target: 目标模式（必须匹配并返回的内容）
            prefix: 前置上下文模式（可选，匹配但跳过）
            suffix: 后置上下文模式（可选，匹配但跳过）
            start_from: 起始行号（None则使用parser.cursor）
            allow_partial: 是否允许suffix匹配失败（仅返回target）
        
        返回:
            Tuple[new_cursor, matched_lines]
            - new_cursor: 匹配结束后的新cursor位置（如果allow_partial=True且在suffix失败时，cursor指向target之后）
            - matched_lines: 仅target部分的内容行列表
        
        抛出:
            PatternNotFoundError: 如果prefix或target未找到（suffix失败时根据allow_partial决定）
            CursorOutOfBoundsError: 如果start_from越界
        
        示例:
            # 匹配标题行（跳过前面的空行，跳过后面的分隔线）
            parser.match_sequence(
                prefix=[parser.skip(1)],  # 跳过可能存在的空行
                target=[parser.any_of("#\\s+Title", "##\\s+Title")],
                suffix=[parser.any_of("---", "##\\s+")]  # 遇到分隔线或新标题结束
            )
        """
        # 确定起始位置
        if start_from is None:
            start_line = self.cursor
        else:
            self._validate_cursor(start_from)
            start_line = start_from
        
        current_cursor = start_line
        
        # 1. 匹配prefix（如果提供）
        if prefix:
            try:
                # prefix匹配但不返回，只需验证存在性
                current_cursor, _ = self._match_at_position(current_cursor, prefix)
            except PatternNotFoundError as e:
                raise PatternNotFoundError(prefix, start_line, self.lines[start_line:start_line+5])
        
        # 2. 匹配target（必须成功）
        try:
            target_start = current_cursor
            
            # 记录target开始前的位置，用于在suffix失败时回退
            before_target_cursor = current_cursor
            
            target_end, matched_line_indices = self._match_at_position(current_cursor, target)
            target_lines = self.lines[target_start:target_end]
        except PatternNotFoundError as e:
            raise PatternNotFoundError(target, current_cursor, self.lines[current_cursor:current_cursor+5])
        
        # 3. 更新cursor到target之后
        current_cursor = target_end
        
        # 4. 匹配suffix（如果提供）
        if suffix:
            try:
                # suffix匹配但不返回，只需验证存在性
                current_cursor, _ = self._match_at_position(current_cursor, suffix)
            except PatternNotFoundError as e:
                if allow_partial:
                    # 允许部分匹配，cursor保持在target之后
                    pass
                else:
                    # 不允许部分匹配，抛出异常
                    # cursor保持在before_target_cursor，让调用者知道从哪里重试
                    self.cursor = before_target_cursor
                    raise PatternNotFoundError(suffix, target_end, self.lines[target_end:target_end+5])
        
        return current_cursor, target_lines


# ============================================================================
# 七、测试案例 - 完整演示
# ============================================================================

if __name__ == "__main__":
    import re
    
    # 测试用的Markdown文本（模拟真实新闻网站）
    sample_markdown = """
# Bitcoin News Aggregator

## Top Stories
Some top content here...

## Latest Bitcoin News

![Bitcoin price chart](https://example.com/btc.png)
[Why is Bitcoin going up today?](https://crypto.news/story1)  
Description of story 1...

3 hours ago

![Mining news](https://example.com/mining.png)  
[Bitcoin mining difficulty increases](https://crypto.news/story2)  
Another description...

14 hours ago

## Other Sections
More content here...
"""
    
    print("=" * 80)
    print("测试案例：从Markdown中提取新闻链接")
    print("=" * 80)
    
    # 1. 创建解析器实例
    parser = MarkdownParser(sample_markdown, strategy="lazy")
    print(f"✓ 创建解析器，总行数: {len(parser.lines)}")
    
    # 2. 使用 extract_between 提取"## Latest Bitcoin News"区域
    print("\n--- 步骤1: 提取 'Latest Bitcoin News' 区域 ---")
    try:
        end_cursor, latest_news_section = parser.extract_between(
            start_marker=[parser.any_of("##\\s+Latest\\s+Bitcoin\\s+News")],
            end_marker=[parser.any_of("##\\s+Other\\s+Sections", "##\\s+")],
            inclusive=False
        )
        print(f"✓ 成功提取区域（{len(latest_news_section)}行）")
        print(f"  新cursor位置: {end_cursor}")
        
        # 显示提取内容（前5行）
        for i, line in enumerate(latest_news_section[:5], 1):
            print(f"    {i}: {line.strip()}")
        if len(latest_news_section) > 5:
            print(f"    ...（共{len(latest_news_section)}行）")
    
    except MarkdownParserError as e:
        print(f"✗ 提取失败: {e}")
        latest_news_section = []
    
    # 3. 创建子解析器处理提取出的区域
    if latest_news_section:
        print("\n--- 步骤2: 在区域内提取新闻块 ---")
        section_parser = MarkdownParser("".join(latest_news_section))
        
        # 使用 match_sequence 匹配每个新闻块
        # 模式结构：图片 + 链接 + 描述 + 时间戳
        news_blocks = []
        current_pos = 0
        
        while current_pos < len(section_parser.lines):
            try:
                new_pos, block_lines = section_parser.match_sequence(
                    prefix=[section_parser.skip_between(0, 2)],  # 跳过0-2行空行
                    target=[section_parser.any_of("!\\[", "\\[")],  # 匹配图片或链接开始
                    suffix=[section_parser.any_of("\\d+\\s+hour", "\\d+\\s+day")],  # 匹配时间戳
                    start_from=current_pos,
                    allow_partial=True  # 允许最后一个块没有时间戳
                )
                
                news_blocks.append(block_lines)
                current_pos = new_pos
                
                # 如果遇到时间戳，说明是完整块；否则可能是末尾不完整的块，停止
                if any("hour" in line or "day" in line for line in block_lines[-2:]):
                    continue
                else:
                    break
            
            except PatternNotFoundError:
                break
        
        print(f"✓ 提取到 {len(news_blocks)} 个新闻块")
        
        # 4. 从每个块中提取链接
        print("\n--- 步骤3: 从每个块中提取链接 ---")
        all_links = []
        for i, block in enumerate(news_blocks, 1):
            block_text = "".join(block)
            # 简单正则提取Markdown链接
            link_matches = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', block_text)
            
            if link_matches:
                title, url = link_matches[0]  # 取第一个链接
                all_links.append((title, url))
                print(f"  块{i}: {title[:50]}... -> {url}")
            else:
                print(f"  块{i}: 未找到链接")
        
        print(f"\n✓ 总计提取 {len(all_links)} 个新闻链接")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)