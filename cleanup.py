#!/usr/bin/env python3
"""
清理脚本 - 删除模拟文件以重新开始

使用场景:
- 重新设计模型时
- 修改模型参数后
- 更改investment_horizon后

使用方法:
    python cleanup.py [investment_horizon]

参数:
    investment_horizon: 可选，'M'(月), 'W'(周), 'd'(日), 'all'(全部)
    默认: 'M'
"""

import os
import sys
import glob

def cleanup_simulation_files(horizon='M'):
    """
    删除指定投资周期的模拟文件
    
    Args:
        horizon: 投资周期，'M', 'W', 'd', 或 'all'
    """
    if horizon == 'all':
        # 删除所有模拟文件
        patterns = [
            'L10_temp_*.pkl',
            '*.pkl'
        ]
    else:
        # 删除特定周期的模拟文件
        horizon_lower = horizon.lower()
        patterns = [
            f'L10_temp_*_{horizon_lower}.pkl',
            f'L10_temp_optimized_{horizon_lower}.pkl',
            f'L10_temp_fixed_{horizon_lower}_*.pkl'
        ]
    
    deleted_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    deleted_files.append(file)
                    print(f"✓ 已删除: {file}")
                except Exception as e:
                    print(f"✗ 删除失败 {file}: {e}")
    
    if not deleted_files:
        print(f"未找到投资周期为 '{horizon}' 的模拟文件")
    else:
        print(f"\n总共删除了 {len(deleted_files)} 个文件")
    
    return deleted_files


def main():
    """主函数"""
    print("=" * 60)
    print("模拟文件清理工具")
    print("=" * 60)
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        horizon = sys.argv[1].upper()
        if horizon not in ['M', 'W', 'D', 'ALL']:
            print(f"错误: 不支持的投资周期 '{horizon}'")
            print("支持的选项: M (月), W (周), D (日), ALL (全部)")
            sys.exit(1)
    else:
        horizon = 'M'  # 默认为月度
    
    print(f"投资周期: {horizon}")
    print()
    
    # 确认操作
    if horizon == 'ALL':
        response = input("警告: 将删除所有模拟文件！是否继续? (y/N): ")
    else:
        response = input(f"将删除投资周期为 '{horizon}' 的模拟文件，是否继续? (y/N): ")
    
    if response.lower() not in ['y', 'yes']:
        print("操作已取消")
        sys.exit(0)
    
    print()
    deleted = cleanup_simulation_files(horizon)
    
    print()
    print("=" * 60)
    print("清理完成！可以重新运行策略进行训练和回测。")
    print("=" * 60)


if __name__ == '__main__':
    main()
