import numpy as np
import tensorflow as tf


class Metrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex

        self.loss_metric = tf.keras.metrics.Mean()
        self.mean_mae_metric = tf.keras.metrics.Mean()
        
        # 用多个Mean指标替代MeanTensor
        self.maes_metrics = [tf.keras.metrics.Mean() for _ in range(len(targets))]

    def update_state(self, loss, mean_mae, mae, nsamples):
        self.loss_metric.update_state(loss, sample_weight=nsamples)
        self.mean_mae_metric.update_state(mean_mae, sample_weight=nsamples)
        
        # 更新每个目标的MAE指标
        for i, mae_metric in enumerate(self.maes_metrics):
            if i < len(mae):
                mae_metric.update_state(mae[i], sample_weight=nsamples)

    def write(self):
        """Write metrics to tf.summary and the Sacred experiment."""
        for key, val in self.result().items():
            tf.summary.scalar(key, val)
            if self.ex is not None:
                if key not in self.ex.current_run.info:
                    self.ex.current_run.info[key] = []
                self.ex.current_run.info[key].append(val)

        if self.ex is not None:
            if f'step_{self.tag}' not in self.ex.current_run.info:
                self.ex.current_run.info[f'step_{self.tag}'] = []
            self.ex.current_run.info[f'step_{self.tag}'].append(tf.summary.experimental.get_step())

    def reset_states(self):
        self.loss_metric.reset_states()
        self.mean_mae_metric.reset_states()
        for mae_metric in self.maes_metrics:
            mae_metric.reset_states()

    def keys(self):
        keys = [f'loss_{self.tag}', f'mean_mae_{self.tag}', f'mean_log_mae_{self.tag}']
        keys.extend([key + '_' + self.tag for key in self.targets])
        return keys

    def result(self):
        result_dict = {}
        result_dict[f'loss_{self.tag}'] = self.loss
        result_dict[f'mean_mae_{self.tag}'] = self.mean_mae
        result_dict[f'mean_log_mae_{self.tag}'] = self.mean_log_mae
        # 添加每个FIA目标的MAE
        for i, key in enumerate(self.targets):
            if i < len(self.maes):
                result_dict[key + '_' + self.tag] = self.maes[i]
        return result_dict

    @property
    def loss(self):
        return self.loss_metric.result().numpy().item()

    @property
    def maes(self):
        return np.array([mae_metric.result().numpy() for mae_metric in self.maes_metrics])

    @property
    def mean_mae(self):
        return self.mean_mae_metric.result().numpy().item()

    @property
    def mean_log_mae(self):
        return np.mean(np.log(self.maes)).item()
    
    def get_fia_metrics(self):
        """获取FIA特定的评估指标"""
        fia_metrics = {}
        for i, target in enumerate(self.targets):
            if i < len(self.maes):
                fia_metrics[f"{target}_mae"] = self.maes[i]
                fia_metrics[f"{target}_mae_{self.tag}"] = self.maes[i]
        return fia_metrics
    
    def print_fia_summary(self):
        """打印FIA评估摘要"""
        print(f"\n=== {self.tag.upper()} FIA 评估摘要 ===")
        for i, target in enumerate(self.targets):
            if i < len(self.maes):
                print(f"{target}: MAE = {self.maes[i]:.6f}")
        print(f"平均MAE: {self.mean_mae:.6f}")
        print(f"平均logMAE: {self.mean_log_mae:.6f}")
        print("=" * 40)
