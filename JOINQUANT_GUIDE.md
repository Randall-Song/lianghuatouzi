# 聚宽平台使用指南

## 快速开始

### 步骤1：上传文件
在聚宽研究平台，上传 `L10_improved_strategy.ipynb` 文件。

### 步骤2：运行策略
按顺序执行所有单元格。第一次运行时间较长（需要准备训练数据和训练模型）。

### 步骤3：查看结果
最后会输出：
- 信息比率
- 策略夏普比率
- 基准ETF夏普比率
- 策略财富曲线图
- 性能对比柱状图

## 关键配置

### 调仓频率
```python
investment_horizon = 'M'  # 月度调仓
# investment_horizon = 'W'  # 周度调仓（可选）
```

如果改为周度调仓，需要同时修改：
```python
number_of_periods_per_year = 52  # 一年52个交易周
```

### 选股数量
```python
optimal_stock_count = 50  # 选择50只股票
```

可以尝试其他值，如：30, 40, 60, 80

### 回测时间段
```python
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2025, 12, 15)
```

## 常见问题

### Q1: 为什么第一次运行很慢？
**A:** 需要：
1. 下载3年的历史数据用于训练
2. 计算所有因子
3. 训练和比较多个模型
4. 运行完整回测

后续运行会使用断点续跑机制，速度会快很多。

### Q2: 如何重新训练模型？
**A:** 代码会自动删除旧的simulation文件。如果需要断点续跑，注释掉这几行：
```python
# if os.path.exists(simulation_file):
#     print(f"删除旧的模拟文件: {simulation_file}")
#     os.remove(simulation_file)
```

### Q3: 信息比率是负数怎么办？
**A:** 可能的原因和解决方法：
1. **模型过拟合**：减少特征数量，增大正则化参数
2. **选股数量不当**：尝试调整 `optimal_stock_count`
3. **调仓频率**：尝试改为周度调仓 `investment_horizon = 'W'`
4. **时间段选择**：某些时间段市场行情可能不适合该策略

### Q4: 出现"因子数据不存在"错误？
**A:** 检查：
1. 日期是否在数据可用范围内
2. 是否有因子数据访问权限
3. 是否使用了正确的指数代码（000852.XSHG）

### Q5: 内存不足怎么办？
**A:** 
1. 减少训练数据的时间跨度（当前是3年）
2. 减少因子数量（修改因子类别筛选条件）
3. 增加特征选择阈值（选择更少的特征）

## 性能优化技巧

### 1. 调整特征选择阈值
```python
# 当前：选择相关性前50%的因子
threshold = correlations.quantile(0.5)

# 更激进：只选择相关性前30%的因子
threshold = correlations.quantile(0.7)
```

### 2. 尝试不同的选股数量
测试不同的 `optimal_stock_count` 值：
- 少量集中：20-30只
- 中等分散：40-60只
- 高度分散：70-100只

### 3. 调整模型复杂度
如果过拟合，增大正则化参数：
```python
'Ridge_alpha5.0': Ridge(alpha=5.0),
'Ridge_alpha10.0': Ridge(alpha=10.0),
```

### 4. 改变因子来源
```python
# 当前：只使用risk和basics类别
all_factors = all_factors.loc[[a in ['risk', 'basics'] 
                 for a in all_factors.loc[:, 'category']], 'factor'].tolist()

# 可以尝试其他类别，如添加quality, growth等
all_factors = all_factors.loc[[a in ['risk', 'basics', 'quality', 'growth'] 
                 for a in all_factors.loc[:, 'category']], 'factor'].tolist()
```

## 解读输出结果

### 模型训练输出
```
因子与收益相关性分析（前10个）:
factor_name1    0.0523
factor_name2    0.0487
...

选择了 15 个因子（相关性 > 0.0234）

Ridge_alpha0.5 平均R2得分: 0.0123 (+/- 0.0045)
Ridge_alpha1.0 平均R2得分: 0.0145 (+/- 0.0039)
...

选择最佳模型: Ridge_alpha1.0 (R2 = 0.0145)
```

**含义：**
- 显示了与收益相关性最高的因子
- 选择了多少个因子参与建模
- 各个模型的交叉验证得分
- 最终选择的最佳模型

### 性能指标输出
```
性能指标:
我的策略夏普    0.8523
基准ETF夏普     0.6234
信息比率        0.3456
```

**含义：**
- **策略夏普比率**：越高越好（通常>1为优秀）
- **基准夏普比率**：基准的风险调整后收益
- **信息比率**：超额收益的质量（>0为跑赢基准，越大越好）

## 进阶用法

### 1. 保存模型供后续使用
```python
import pickle

# 保存模型和特征
with open('my_best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': my_model,
        'features': selected_feature_names,
        'stock_count': optimal_stock_count
    }, f)

# 加载模型
with open('my_best_model.pkl', 'rb') as f:
    saved = pickle.load(f)
    my_model = saved['model']
    selected_feature_names = saved['features']
```

### 2. 分析单个因子的重要性
```python
# 查看所有因子的系数
if hasattr(my_model, 'coef_'):
    coef_df = pd.DataFrame({
        'factor': selected_feature_names,
        'coefficient': my_model.coef_
    })
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    print(coef_df)
```

### 3. 逐月性能分析
```python
# 计算每月收益
monthly_returns = combined_df_ret.loc[:, '我的策略']
print("月度收益统计:")
print(f"平均月收益: {monthly_returns.mean()*100:.2f}%")
print(f"最大月收益: {monthly_returns.max()*100:.2f}%")
print(f"最小月收益: {monthly_returns.min()*100:.2f}%")
print(f"胜率: {(monthly_returns > 0).sum() / len(monthly_returns)*100:.1f}%")
```

## 注意事项

1. **数据泄露**：确保训练数据严格在start_date之前
2. **前视偏差**：cal_portfolio_weight_series中不能使用未来数据
3. **交易成本**：已考虑0.1%的双边成本
4. **ST和停牌**：已自动排除
5. **因子标准化**：使用了鲁棒的标准化方法

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

## 许可证

本项目仅供学习和研究使用。
