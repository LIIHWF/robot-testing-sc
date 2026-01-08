#!/usr/bin/env python3
"""
信号录制 Wrapper

这个 wrapper 会自动将环境的信号添加到 info 字典中，以便在录制时保存信号。

使用方法:
    在创建环境时包装环境：
    
    from signal_recording_wrapper import SignalRecordingWrapper
    env = gym.make("your_env_name")
    env = SignalRecordingWrapper(env)
"""

import gymnasium as gym
from typing import Dict, Any


class SignalRecordingWrapper(gym.Wrapper):
    """
    自动将环境的信号添加到 info 字典中的 wrapper。
    
    这个 wrapper 会在每个 step 中调用环境的 get_subtask_term_signals() 方法
    并将结果添加到 info 字典的 'signals' 键中。
    """
    
    def __init__(self, env):
        """
        初始化 wrapper。
        
        Args:
            env: 要包装的环境
        """
        super().__init__(env)
        self._check_signal_method()
    
    def _check_signal_method(self):
        """检查环境是否有 get_subtask_term_signals 方法"""
        if not hasattr(self.env, 'get_subtask_term_signals'):
            # 尝试访问底层环境
            if hasattr(self.env, 'env'):
                if hasattr(self.env.env, 'get_subtask_term_signals'):
                    return
            print("警告: 环境没有 get_subtask_term_signals() 方法，信号将不会被记录")
    
    def _get_signals(self) -> Dict[str, int]:
        """
        获取环境的信号。
        
        Returns:
            信号字典，键为信号名称，值为 0 或 1
        """
        # 尝试从当前环境获取
        if hasattr(self.env, 'get_subtask_term_signals'):
            try:
                return self.env.get_subtask_term_signals()
            except Exception as e:
                print(f"警告: 获取信号时出错: {e}")
                return {}
        
        # 尝试从底层环境获取
        if hasattr(self.env, 'env'):
            if hasattr(self.env.env, 'get_subtask_term_signals'):
                try:
                    return self.env.env.get_subtask_term_signals()
                except Exception as e:
                    print(f"警告: 从底层环境获取信号时出错: {e}")
                    return {}
        
        return {}
    
    def step(self, action):
        """
        执行一步并添加信号到 info。
        
        Args:
            action: 动作
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 获取信号并添加到 info
        signals = self._get_signals()
        if signals:
            # 处理向量化环境的情况
            # info 中的值可能是列表（每个环境一个值）
            if isinstance(info, dict):
                # 检查是否是向量化环境
                if 'success' in info and isinstance(info['success'], list):
                    # 向量化环境：为每个环境创建信号列表
                    n_envs = len(info['success'])
                    signals_list = {}
                    for signal_name, signal_value in signals.items():
                        # 如果信号值已经是列表，直接使用；否则为每个环境复制
                        if isinstance(signal_value, (list, tuple)):
                            signals_list[signal_name] = list(signal_value)
                        else:
                            signals_list[signal_name] = [int(signal_value)] * n_envs
                    info['signals'] = signals_list
                else:
                    # 单个环境：直接添加信号字典
                    info['signals'] = {k: int(v) for k, v in signals.items()}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """
        重置环境。
        
        Args:
            **kwargs: 传递给底层环境的参数
            
        Returns:
            (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        
        # 在重置时也获取信号（虽然通常重置时信号应该都是 0）
        signals = self._get_signals()
        if signals:
            if isinstance(info, dict):
                if 'success' in info and isinstance(info['success'], list):
                    n_envs = len(info['success'])
                    signals_list = {}
                    for signal_name, signal_value in signals.items():
                        if isinstance(signal_value, (list, tuple)):
                            signals_list[signal_name] = list(signal_value)
                        else:
                            signals_list[signal_name] = [int(signal_value)] * n_envs
                    info['signals'] = signals_list
                else:
                    info['signals'] = {k: int(v) for k, v in signals.items()}
        
        return obs, info


# 示例使用
if __name__ == "__main__":
    import gymnasium as gym
    
    # 示例：包装环境
    # env = gym.make("your_env_name")
    # env = SignalRecordingWrapper(env)
    
    print("SignalRecordingWrapper 已定义")
    print("使用方法:")
    print("  from signal_recording_wrapper import SignalRecordingWrapper")
    print("  env = gym.make('your_env_name')")
    print("  env = SignalRecordingWrapper(env)")

