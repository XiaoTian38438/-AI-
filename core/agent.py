"""智能体基类 - 支持LLM增强和工具调用

赛事要求：系统中包含1个以上智能体，多智能体协同
"""
from core.llm import call as llm_call


class Agent:
    """智能体基类"""

    def __init__(self, name, description='', capabilities=None):
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self._tools = {}

    def register_tool(self, name, func, description=''):
        """注册工具函数"""
        self._tools[name] = {'func': func, 'description': description}

    def use_tool(self, name, *args, **kwargs):
        """使用工具"""
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册")
        return self._tools[name]['func'](*args, **kwargs)

    def run(self, context):
        """执行智能体任务（子类实现）"""
        raise NotImplementedError

    def __repr__(self):
        return f"Agent({self.name})"


class LLMAgent(Agent):
    """LLM增强智能体 - 使用大模型进行推理决策

    赛事评分项：
    - 多模态调用(5%): 包含一个以上大模型调用
    - 提示词设计质量(5%): 有效地与大模型交互
    """

    def __init__(self, name, description='', system_prompt='', capabilities=None):
        super().__init__(name, description, capabilities)
        self.system_prompt = system_prompt

    def think(self, prompt, fallback=None):
        """调用LLM进行推理"""
        return llm_call(prompt, system_prompt=self.system_prompt, fallback=fallback)

    def run(self, context):
        raise NotImplementedError


class ToolAgent(Agent):
    """工具智能体 - 执行具体的数据采集或分析任务"""

    def __init__(self, name, description='', func=None, capabilities=None):
        super().__init__(name, description, capabilities)
        self._func = func

    def run(self, context):
        if self._func:
            return self._func(context)
        return None
