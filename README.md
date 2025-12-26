# 量化投资策略 - 多因子模型

## 项目说明

本项目实现了一个基于多因子的量化投资策略，旨在最大化信息比率（Information Ratio）。

## 主要改进

### 1. 模型改进
- **从 Ridge 回归改为 Gradient Boosting Regressor**
  - Ridge回归只能捕捉线性关系
  - Gradient Boosting可以捕捉非线性关系和因子之间的交互作用
  - 使用Huber损失函数，对异常值更加鲁棒

### 2. 超参数优化
- `n_estimators=150`: 更多的树来学习复杂模式
- `learning_rate=0.03`: 较低的学习率提高泛化能力
- `max_depth=5`: 适度的树深度平衡复杂度和泛化
- `min_samples_split=30, min_samples_leaf=15`: 防止过拟合
- `subsample=0.7`: 使用70%样本训练每棵树，提高泛化
- `max_features='sqrt'`: 减少特征相关性，防止过拟合

### 3. 训练数据扩展
- 从3年训练数据扩展到4年
- 更多的训练样本有助于模型学习更稳定的模式
- 所有训练数据都在start_date之前，符合要求

### 4. 投资组合优化
- **股票数量优化**: 从50只减少到40只
  - 更集中的投资组合，选择最优质的股票
  - 减少持仓分散化带来的稀释效应
  
- **加权方式改进**: 从等权改为预测值加权
  - 给予预测收益更高的股票更大权重
  - 更好地利用模型的预测信息

## 使用方法

### 1. 首次运行（训练新模型）
```python
# 删除已有的模型和模拟文件
import os
if os.path.exists('my_model.pkl'):
    os.remove('my_model.pkl')
if os.path.exists('L10_temp_fixed_m_basicsrisk.pkl'):
    os.remove('L10_temp_fixed_m_basicsrisk.pkl')

# 运行策略
python strategy.py
```

### 2. 继续运行（使用已训练模型）
```python
# 直接运行，会自动加载已保存的模型
python strategy.py
```

### 3. 断点续跑
- 策略支持断点续跑功能
- 模拟进度保存在 `L10_temp_fixed_m_basicsrisk.pkl`
- 如果中断，重新运行会从上次中断处继续

### 4. 重新设计模型
根据要求，重新设计模型时需要删除本地的simulation_file：
```python
import os
if os.path.exists('L10_temp_fixed_m_basicsrisk.pkl'):
    os.remove('L10_temp_fixed_m_basicsrisk.pkl')
```

## 配置参数

在 `strategy.py` 中可以调整以下参数：

```python
start_date = datetime.date(2020, 1, 1)  # 回测开始日期
end_date = datetime.date(2025, 12, 15)  # 回测结束日期
investment_horizon = 'M'                # 'M'=月度, 'W'=周度, 'D'=日度
number_of_periods_per_year = 12         # 12=月度, 52=周度, 252=日度
```

## 约束条件

1. **不能修改 `cal_portfolio_weight_series(decision_date)` 函数的接口**
   - 函数签名保持为: `cal_portfolio_weight_series(decision_date, old_portfolio_weight_series)`
   - 返回值为包含股票代码和权重的Series

2. **训练数据必须是start_date之前的数据**
   - 当前使用 start_date 前4年的数据进行训练
   - 确保不存在未来数据泄露

3. **investment_horizon 可选 W 或 M**
   - 当前设置为 'M' (月度调仓)
   - 可根据需要改为 'W' (周度调仓)

4. **在聚宽环境中运行，不改变输出数据的格式**
   - 保持与原始代码相同的输出格式
   - 输出信息比率等关键指标

## 输出结果

程序运行后会输出：
- 策略夏普比率
- 基准ETF夏普比率
- **信息比率**（关键指标）

## 文件说明

- `strategy.py`: 主策略文件
- `my_model.pkl`: 训练好的模型文件
- `L10_temp_fixed_m_basicsrisk.pkl`: 回测断点续跑文件
- `L10_多因子策略_比赛.html`: 原始课件（包含问题说明和基准代码）

## 技术栈

- pandas: 数据处理
- numpy: 数值计算
- scikit-learn: 机器学习模型
- jqdata/jqfactor: 聚宽数据接口
- pickle: 模型序列化

## 性能目标

目标：最大化信息比率（Information Ratio）

信息比率 = (策略收益率 - 基准收益率) / 超额收益波动率

通过改进模型和投资组合构建方法，预期获得比基准Ridge模型（信息比率≈0.27）更高的信息比率。
