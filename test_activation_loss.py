#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试激活函数和损失函数"""

import numpy as np
from ops.activations import Identity, Sigmoid, ReLU
from ops.losses import SoftmaxCrossEntropyLoss, CrossEntropyLoss

def test_identity():
    print("测试 Identity 激活函数...")
    identity = Identity()
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for x in test_values:
        y = identity.forward(x)
        dy = identity.derivative(x)
        print(f"  Identity({x:.2f}) = {y:.2f}, derivative = {dy:.2f}")
        assert abs(y - x) < 1e-6, f"Identity forward error: {y} != {x}"
        assert abs(dy - 1.0) < 1e-6, f"Identity derivative error: {dy} != 1.0"
    print("  ✓ Identity 测试通过\n")

def test_softmax_cross_entropy():
    print("测试 SoftmaxCrossEntropyLoss...")
    loss_fn = SoftmaxCrossEntropyLoss()
    
    # 测试1: 简单二分类
    pred = np.array([1.0, 0.0])
    target = np.array([1, 0])  # one-hot
    loss = loss_fn.forward(pred, target)
    grad = loss_fn.backward(pred, target)
    print(f"  测试1 - 预测: {pred}, 目标: {target}")
    print(f"    损失: {loss:.6f}, 梯度: {grad}")
    
    # 测试2: 索引标签
    pred2 = np.array([2.0, 1.0, 0.0])
    target2 = 0  # 类别0
    loss2 = loss_fn.forward(pred2, target2)
    grad2 = loss_fn.backward(pred2, target2)
    print(f"  测试2 - 预测: {pred2}, 目标索引: {target2}")
    print(f"    损失: {loss2:.6f}, 梯度: {grad2}")
    
    # 验证梯度数值稳定性
    pred3 = np.array([1000.0, 0.0])  # 大数值测试
    target3 = np.array([1, 0])
    loss3 = loss_fn.forward(pred3, target3)
    grad3 = loss_fn.backward(pred3, target3)
    print(f"  测试3 - 大数值预测: {pred3}")
    print(f"    损失: {loss3:.6f}, 梯度: {grad3}")
    
    # 验证梯度之和为0
    for grad in [grad, grad2, grad3]:
        grad_sum = np.sum(grad)
        assert abs(grad_sum) < 1e-6, f"梯度之和不为0: {grad_sum}"
    
    print("  ✓ SoftmaxCrossEntropyLoss 测试通过\n")

def test_cross_entropy():
    print("测试 CrossEntropyLoss...")
    loss_fn = CrossEntropyLoss()
    
    # 二分类测试
    pred = np.array([0.9, 0.1, 0.7, 0.3])
    target = np.array([1, 0, 1, 0])
    loss = loss_fn.forward(pred, target)
    grad = loss_fn.backward(pred, target)
    print(f"  预测: {pred}, 目标: {target}")
    print(f"    损失: {loss:.6f}, 梯度: {grad}")
    
    # 验证损失非负
    assert loss >= 0, f"损失为负: {loss}"
    print("  ✓ CrossEntropyLoss 测试通过\n")

if __name__ == "__main__":
    test_identity()
    test_softmax_cross_entropy()
    test_cross_entropy()
    print("所有测试通过！")