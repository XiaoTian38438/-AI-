# engine/trainer.py
import numpy as np
from engine.optimizers import clip_grad_norm
from engine.scheduler import StepLR, ExponentialLR

try:
    import cupy as cp
    _HAS_CP = True
except ImportError:
    cp = None
    _HAS_CP = False


class Trainer:
    def __init__(self, network, loss_fn, optimizer, scheduler=None, grad_clip=None):
        self.net = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.train_losses = []
        self.val_losses = []

    def train(self, X, y, epochs, batch_size=1, X_val=None, y_val=None,
              early_stopping_patience=None, verbose=True):
        n = X.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0

        # 确保向量化模型已构建
        self.net._ensure_vm()
        vm = self.net._vm
        xp = vm.xp

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_indices = indices[start:end]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                bs = len(X_batch)

                # 清零向量化梯度
                for layer in vm.layers:
                    layer.W_grad[:] = 0
                    layer.b_grad[:] = 0

                # 前向 + loss + 反向（全在 GPU/CPU 矩阵上）
                preds = vm.forward(X_batch, training=True)

                if hasattr(self.loss_fn, 'forward_batch'):
                    batch_loss, grad = self.loss_fn.forward_batch(preds, y_batch)
                else:
                    preds_cpu = np.asarray(preds) if xp is not np else preds
                    batch_loss = 0.0
                    grad = xp.zeros_like(preds)
                    for i in range(bs):
                        batch_loss += self.loss_fn.forward(preds_cpu[i], y_batch[i])
                        grad[i] = self.loss_fn.backward(preds_cpu[i], y_batch[i])

                vm.backward(grad)

                # 梯度平均
                for layer in vm.layers:
                    layer.W_grad /= bs
                    layer.b_grad /= bs

                # 梯度裁剪
                if self.grad_clip is not None:
                    clip_grad_norm(self.net, self.grad_clip)

                # 向量化优化器更新
                if hasattr(self.optimizer, 'step_vm'):
                    self.optimizer.step_vm(vm.layers)
                else:
                    # 降级：同步到节点/边再更新
                    vm.sync_grads_to_nodes()
                    self.optimizer.step()
                    vm.sync_weights_from_nodes()

                epoch_loss += batch_loss / bs
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)

            # 验证
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                self.val_losses.append(val_loss)
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

            if self.scheduler is not None:
                if isinstance(self.scheduler, (StepLR, ExponentialLR)):
                    self.scheduler.step()
                else:
                    if val_loss is not None:
                        self.scheduler.step(val_loss)

        # 训练结束后同步权重到节点/边（兼容 save/load）
        vm.sync_weights_to_nodes_full()

        if hasattr(self.optimizer, 'lr'):
            print(f"Final learning rate: {self.optimizer.lr:.6f}")

    def evaluate(self, X, y):
        vm = self.net._vm
        if vm is not None:
            preds = vm.forward(X, training=False)
            if self.net.use_gpu:
                preds_cpu = cp.asnumpy(preds)
            else:
                preds_cpu = np.asarray(preds)
        else:
            preds_cpu = np.array([self.net.forward(x) for x in X])

        total_loss = 0.0
        for i in range(len(X)):
            total_loss += self.loss_fn.forward(preds_cpu[i], y[i])
        return total_loss / len(X)

    def predict(self, X):
        vm = self.net._vm
        if vm is not None:
            preds = vm.forward(X, training=False)
            if self.net.use_gpu:
                return cp.asnumpy(preds)
            return np.asarray(preds)
        return np.array([self.net.forward(x) for x in X])
