"""
数据工程阶段 - 数据验证与可视化
验证面板数据的完整性和特征质量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
print("=" * 60)
print("数据工程验证报告")
print("=" * 60)

df = pd.read_csv("df_train_panel.csv")
df_valid = pd.read_csv("df_train_valid.csv")
df_modern = pd.read_csv("df_train_modern.csv")

# ==================== 1. 数据集概览 ====================
print("\n" + "=" * 60)
print("1. 数据集概览")
print("=" * 60)

print(f"\n📊 完整面板数据 (df_train_panel.csv):")
print(f"   - 总行数: {len(df):,}")
print(f"   - 总列数: {len(df.columns)}")
print(f"   - 年份范围: {df['Year'].min()} - {df['Year'].max()}")
print(f"   - 国家数: {df['NOC'].nunique()}")
print(f"   - 奥运届数: {df['Year'].nunique()}")

print(f"\n📊 有效训练数据 (df_train_valid.csv):")
print(f"   - 总行数: {len(df_valid):,} (有Lag1特征)")

print(f"\n📊 现代数据 (df_train_modern.csv):")
print(f"   - 总行数: {len(df_modern):,} (1984年后)")

# ==================== 2. 特征完整性 ====================
print("\n" + "=" * 60)
print("2. 特征完整性检查")
print("=" * 60)

print("\n缺失值统计:")
null_stats = df.isnull().sum()
null_stats = null_stats[null_stats > 0]
if len(null_stats) > 0:
    for col, count in null_stats.items():
        pct = count / len(df) * 100
        print(f"   - {col}: {count:,} ({pct:.1f}%)")
else:
    print("   ✅ 无缺失值")

# ==================== 3. 特征分布 ====================
print("\n" + "=" * 60)
print("3. 关键特征统计")
print("=" * 60)

key_features = ['Target', 'Squad_Size', 'EWMA_Score', 'Efficiency', 'Events_Participated']
print("\n" + df[key_features].describe().round(2).to_string())

# ==================== 4. 东道主效应分析 ====================
print("\n" + "=" * 60)
print("4. 东道主效应分析")
print("=" * 60)

hosts = df[df['Is_Host'] == 1][['NOC', 'Year', 'Target', 'Lag1_Medals']].dropna().copy()
hosts['Boost'] = hosts['Target'] - hosts['Lag1_Medals']
hosts['Boost_Pct'] = (hosts['Boost'] / hosts['Lag1_Medals'] * 100).round(1)

print("\n东道主奖牌提升:")
print(hosts.sort_values('Year').tail(15).to_string(index=False))

print(f"\n平均东道主提升: {hosts['Boost'].mean():.1f} 块奖牌")
print(f"平均东道主提升率: {hosts['Boost_Pct'].mean():.1f}%")

# ==================== 5. 地区分布 ====================
print("\n" + "=" * 60)
print("5. 地区分布")
print("=" * 60)

region_stats = df.groupby('Region').agg({
    'NOC': 'nunique',
    'Target': ['count', 'mean', 'sum']
}).round(1)
region_stats.columns = ['国家数', '记录数', '平均奖牌', '总奖牌']
print(region_stats.sort_values('总奖牌', ascending=False).to_string())

# ==================== 6. 可视化 ====================
print("\n" + "=" * 60)
print("6. 生成可视化图表...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Olympic Medal Prediction - Data Engineering Summary', fontsize=14, fontweight='bold')

# 6.1 奖牌分布
ax1 = axes[0, 0]
df['Target'].hist(bins=50, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Total Medals')
ax1.set_ylabel('Frequency')
ax1.set_title('Medal Distribution')
ax1.axvline(df['Target'].median(), color='red', linestyle='--', label=f'Median: {df["Target"].median():.0f}')
ax1.legend()

# 6.2 Squad Size vs Medals
ax2 = axes[0, 1]
df_plot = df.dropna(subset=['Squad_Size', 'Target'])
ax2.scatter(df_plot['Squad_Size'], df_plot['Target'], alpha=0.5, s=20, c='steelblue')
ax2.set_xlabel('Squad Size')
ax2.set_ylabel('Total Medals')
ax2.set_title('Squad Size vs Medals')
# 添加趋势线
z = np.polyfit(df_plot['Squad_Size'], df_plot['Target'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_plot['Squad_Size'].min(), df_plot['Squad_Size'].max(), 100)
ax2.plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
ax2.legend()

# 6.3 EWMA趋势 - Top 5 Countries
ax3 = axes[0, 2]
top_countries = df.groupby('NOC')['Target'].sum().nlargest(5).index.tolist()
for country in top_countries:
    country_data = df[df['NOC'] == country].sort_values('Year')
    ax3.plot(country_data['Year'], country_data['EWMA_Score'], marker='o', markersize=3, label=country)
ax3.set_xlabel('Year')
ax3.set_ylabel('EWMA Score')
ax3.set_title('EWMA Trend - Top 5 Countries')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 6.4 东道主效应
ax4 = axes[1, 0]
host_effect = df[df['Is_Host'] == 1].dropna(subset=['Lag1_Medals'])
boost = host_effect['Target'] - host_effect['Lag1_Medals']
colors = ['green' if b > 0 else 'red' for b in boost]
ax4.bar(range(len(boost)), boost, color=colors, alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axhline(y=boost.mean(), color='blue', linestyle='--', label=f'Avg: {boost.mean():.1f}')
ax4.set_xlabel('Host Events')
ax4.set_ylabel('Medal Boost')
ax4.set_title('Host Country Medal Boost')
ax4.legend()

# 6.5 地区分布
ax5 = axes[1, 1]
region_medals = df.groupby('Region')['Target'].sum().sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(region_medals)))
region_medals.plot(kind='barh', ax=ax5, color=colors)
ax5.set_xlabel('Total Medals')
ax5.set_title('Total Medals by Region')

# 6.6 特征相关性热图
ax6 = axes[1, 2]
corr_features = ['Target', 'Lag1_Medals', 'EWMA_Score', 'Squad_Size', 'Events_Participated', 'Is_Host']
corr_data = df[corr_features].dropna()
corr_matrix = corr_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax6, 
            fmt='.2f', square=True, linewidths=0.5)
ax6.set_title('Feature Correlation')

plt.tight_layout()
plt.savefig('data_engineering_summary.png', dpi=150, bbox_inches='tight')
print("   ✅ 已保存: data_engineering_summary.png")

# ==================== 7. 顶级国家详细数据 ====================
print("\n" + "=" * 60)
print("7. 顶级国家2024年数据")
print("=" * 60)

top_2024 = df[df['Year'] == 2024].nlargest(15, 'Target')
print(top_2024[['NOC', 'Target', 'Gold', 'Squad_Size', 'EWMA_Score', 'Lag1_Medals', 'Is_Host']].to_string(index=False))

# ==================== 8. 特征工程总结 ====================
print("\n" + "=" * 60)
print("8. 特征工程总结")
print("=" * 60)

feature_summary = """
构建的特征列表:

【惯性特征 (Momentum)】
  1. Lag1_Medals     - 上一届奖牌数
  2. Lag2_Medals     - 上上届奖牌数  
  3. Lag3_Medals     - 前三届奖牌数
  4. Weighted_Avg_3  - 过去三届加权平均 (权重: 0.5, 0.3, 0.2)
  5. EWMA_Score      - 指数加权移动平均 (α=0.4)
  6. Lag1_EWMA       - 上一届EWMA分数
  7. Momentum        - 趋势指标 (本届-上届)

【投入特征 (Investment)】
  8. Squad_Size          - 参赛运动员人数
  9. Lag1_Squad_Size     - 上届参赛人数
  10. Events_Participated - 参加项目数
  11. Sports_Participated - 参加大项数
  12. Female_Ratio        - 女性运动员比例

【效率特征 (Efficiency)】
  13. Efficiency      - 奖牌效率 (Medals/Squad_Size)
  14. Lag1_Efficiency - 上届奖牌效率
  15. Event_Coverage  - 项目覆盖率
  16. Gold_Ratio      - 金牌占比

【东道主特征 (Host Effect)】
  17. Is_Host             - 是否当届东道主
  18. Is_Next_Host        - 是否下届东道主
  19. Time_Since_Last_Host - 距离上次举办年数

【环境特征 (Context)】
  20. Total_Events    - 当届总项目数
  21. Region          - 地区分类 (Ex-USSR, Europe, Asia, Americas, Oceania, Africa, Other)
  22. Is_Post_1992    - 是否1992年后

【特殊标记】
  23. Is_Boycott_Year   - 是否抵制年份 (1980/1984)
  24. Historical_Weight - 历史数据权重 (1992前为0.7)
"""
print(feature_summary)

print("\n" + "=" * 60)
print("✅ 数据验证完成!")
print("=" * 60)
