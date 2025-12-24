# 多因子量化策略 - 优化版

## 项目概述

本项目实现了一个基于多因子的量化投资策略，目标是最大化信息比率（Information Ratio）。策略使用机器学习模型对股票收益进行预测，并构建优化的投资组合。

> 🚀 **新手？** 查看 [QUICKSTART.md](QUICKSTART.md) 5分钟快速上手！

## 核心改进

相比原始的简单Ridge回归模型，本优化版本实现了以下改进：

### 1. 多模型集成（Ensemble）
- **Ridge回归**: 基准线性模型，提供稳定的预测
- **Lasso回归**: 通过L1正则化进行特征选择
- **ElasticNet**: 结合L1和L2正则化的优势
- **Random Forest**: 捕捉非线性关系和特征交互
- **Gradient Boosting**: 强大的梯度提升模型

集成权重设置：
- Ridge: 15%
- Lasso: 15%
- ElasticNet: 15%
- Random Forest: 25%
- Gradient Boosting: 30%

### 2. 改进的特征处理
- **标准化处理**: 对所有因子进行标准化
- **极端值处理**: 使用对数变换处理极端值
- **Winsorization**: 1%和99%分位数截尾
- **缺失值处理**: 智能填充缺失值

### 3. 模型训练策略
- 使用start_date之前3年的历史数据进行训练
- 避免前视偏差（Look-ahead Bias）
- 支持断点续训功能

## 使用说明

### 环境要求
- Python 3.x
- 聚宽（JoinQuant）量化平台环境
- 依赖库：pandas, numpy, scikit-learn, tqdm

### 配置参数

在 `strategy.py` 中可以修改以下参数：

```python
start_date = datetime.date(2020, 1, 1)  # 回测开始日期
end_date = datetime.date(2025, 12, 15)  # 回测结束日期
investment_horizon = 'M'  # 投资周期: 'M'(月), 'W'(周), 'd'(日)
```

### 运行策略

#### 在聚宽平台运行

1. 将 `strategy.py` 的内容复制到聚宽研究环境或策略编辑器
2. 根据需要调整配置参数
3. 运行脚本

#### 重新训练模型

如果需要重新设计模型或修改训练参数：

```python
# 1. 删除本地模拟文件（重要！）
import os
simulation_file = "L10_temp_optimized_m.pkl"  # 根据investment_horizon调整
if os.path.exists(simulation_file):
    os.remove(simulation_file)
    print(f"已删除 {simulation_file}")

# 2. 重新运行 strategy.py
```

### 文件说明

- `strategy.py`: 主策略文件，包含完整的策略实现
- `L10_多因子策略_比赛.html`: 原始Jupyter Notebook的HTML版本
- `extracted_code.py`: 从HTML提取的代码（用于参考）
- `L10_temp_optimized_*.pkl`: 断点续跑文件（自动生成）

## 核心函数说明

### cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)

**策略核心函数，不可修改！**

功能：
- 在decision_date收盘后决定下一天要买的投资组合
- 不能使用decision_date之后的任何信息
- 返回投资组合权重序列

约束：
- 只能使用decision_date及之前的数据
- 函数签名不能改变

### train_improved_model()

模型训练函数：
- 使用start_date之前3年的数据
- 训练5个不同的机器学习模型
- 返回集成模型

### simulate_wealth_process(start_date, end_date)

回测模拟函数：
- 模拟策略在指定时间段的表现
- 支持断点续跑
- 返回财富过程和配置字典

## 性能指标

策略评估关键指标：

1. **信息比率 (Information Ratio)**
   - 定义: 超额收益 / 跟踪误差
   - 目标: 最大化此指标

2. **夏普比率 (Sharpe Ratio)**
   - 评估风险调整后的收益

3. **年化收益率**
   - 策略的年化回报

## 注意事项

1. **数据要求**: 训练数据必须是start_date之前的数据
2. **断点续跑**: 修改模型后需要删除simulation_file
3. **投资周期**: investment_horizon可选 'M'(月) 或 'W'(周)
4. **平台兼容**: 确保在聚宽环境中运行
5. **因子类别**: 当前使用'risk'和'basics'类别的因子

## 进一步优化方向

1. **超参数调优**: 使用网格搜索或贝叶斯优化
2. **因子扩展**: 增加更多因子类别（如growth, value等）
3. **动态选股**: 调整选股数量（当前固定50只）
4. **风险管理**: 添加止损、仓位控制等机制
5. **集成权重优化**: 基于验证集IC优化各模型权重

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
