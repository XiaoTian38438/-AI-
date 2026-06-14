"""工作流引擎 - 支持DAG调度、条件分支、多智能体协同

赛事要求：
- 基于工作流的方式开发AI智能体应用（10%）
- 工作流必须包含10个节点以上
- 工作流中必须包含基于大模型输出的判断节点
"""
import time
from enum import Enum


class NodeType(Enum):
    INPUT = 'input'
    DATA_FETCH = 'data_fetch'
    AGENT = 'agent'
    DECISION = 'decision'
    MERGE = 'merge'
    VALIDATOR = 'validator'
    OUTPUT = 'output'


class WorkflowNode:
    """工作流节点"""

    def __init__(self, name, node_type, description='', handler=None):
        self.name = name
        self.node_type = node_type
        self.description = description
        self.handler = handler
        self.branches = []          # [(condition_func, target_name, label), ...]
        self.default_next = None
        self.status = 'pending'
        self.result = None

    def execute(self, context):
        self.status = 'running'
        start = time.time()
        try:
            self.result = self.handler(context) if self.handler else None
            self.status = 'success'
            return {'result': self.result, 'elapsed': round(time.time() - start, 3), 'status': 'success'}
        except Exception as e:
            self.status = 'error'
            self.result = None
            return {'result': None, 'error': str(e), 'elapsed': round(time.time() - start, 3), 'status': 'error'}

    def add_branch(self, condition, target, label=''):
        self.branches.append((condition, target, label))

    def get_next(self, context):
        for condition, target, label in self.branches:
            try:
                if condition(context):
                    return target
            except Exception:
                continue
        return self.default_next


class Workflow:
    """工作流引擎"""

    def __init__(self, name, description=''):
        self.name = name
        self.description = description
        self.nodes = {}             # name -> WorkflowNode
        self.start_node = None
        self.execution_log = []

    def add_node(self, name, node_type, description='', handler=None):
        node = WorkflowNode(name, node_type, description, handler)
        self.nodes[name] = node
        if self.start_node is None and node_type == NodeType.INPUT:
            self.start_node = name
        return node

    def connect(self, from_name, to_name, condition=None, label=''):
        if from_name not in self.nodes:
            raise ValueError(f"节点 '{from_name}' 不存在")
        if to_name not in self.nodes:
            raise ValueError(f"节点 '{to_name}' 不存在")
        node = self.nodes[from_name]
        if condition:
            node.add_branch(condition, to_name, label)
        else:
            node.default_next = to_name
        return self

    def run(self, context=None):
        if context is None:
            context = {}

        context.setdefault('_log', [])
        context.setdefault('_data', {})
        self.execution_log = []

        # 重置状态
        for node in self.nodes.values():
            node.status = 'pending'
            node.result = None

        current = self.start_node
        if not current:
            raise ValueError("未设置起始节点")

        while current:
            node = self.nodes[current]
            result = node.execute(context)

            log_entry = {
                'node': current,
                'type': node.node_type.value,
                'status': result['status'],
                'elapsed': result.get('elapsed', 0),
                'description': node.description,
            }
            if result.get('error'):
                log_entry['error'] = result['error']
            self.execution_log.append(log_entry)
            context['_log'].append(log_entry)

            if result['status'] == 'error' and node.node_type != NodeType.DECISION:
                break

            current = node.get_next(context)

        return context

    def get_mermaid(self):
        """生成 Mermaid 流程图代码"""
        lines = ['graph TD']
        for name, node in self.nodes.items():
            shape = {
                NodeType.INPUT: ('([', '])'),
                NodeType.OUTPUT: ('([', '])'),
                NodeType.DECISION: ('{', '}'),
                NodeType.VALIDATOR: ('[[', ']]'),
            }.get(node.node_type, ('[', ']'))
            lines.append(f'    {name}{shape[0]}"{node.description}"{shape[1]}')

        for name, node in self.nodes.items():
            if node.default_next:
                lines.append(f'    {name} --> {node.default_next}')
            for cond, target, label in node.branches:
                arrow = f' -->|"{label}"| {target}' if label else f' --> {target}'
                lines.append(f'    {name}{arrow}')

        # 添加样式
        lines.append('')
        styles = {
            'input': 'fill:#4CAF50,color:#fff',
            'output': 'fill:#2196F3,color:#fff',
            'decision': 'fill:#FF9800,color:#fff',
            'agent': 'fill:#9C27B0,color:#fff',
            'data': 'fill:#00BCD4,color:#fff',
            'validator': 'fill:#F44336,color:#fff',
            'merge': 'fill:#607D8B,color:#fff',
        }
        type_to_style = {
            NodeType.INPUT: 'input', NodeType.OUTPUT: 'output',
            NodeType.DECISION: 'decision', NodeType.AGENT: 'agent',
            NodeType.DATA_FETCH: 'data', NodeType.VALIDATOR: 'validator',
            NodeType.MERGE: 'merge',
        }
        for name, node in self.nodes.items():
            style_key = type_to_style.get(node.node_type)
            if style_key:
                lines.append(f'    style {name} {styles[style_key]}')

        return '\n'.join(lines)

    def summary(self):
        return {
            'name': self.name,
            'description': self.description,
            'node_count': len(self.nodes),
            'nodes': {n: {'type': nd.node_type.value, 'desc': nd.description}
                      for n, nd in self.nodes.items()},
        }
