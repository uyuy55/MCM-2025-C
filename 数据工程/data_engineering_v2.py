"""
数据工程阶段 V2：构建面板数据 (Panel Data)
整合 medal_counts.csv, hosts.csv, athletes.csv 和 programs.csv
构建训练数据集 df_train，每一行是一个"国家-年份"

关键修复：统一NOC代码匹配
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 加载原始数据 ====================
print("=" * 60)
print("1. 加载原始数据")
print("=" * 60)

data_path = Path("2025_Problem_C_Data")

# 加载奖牌数据
df_medals = pd.read_csv(data_path / "summerOly_medal_counts.csv")
print(f"奖牌数据: {df_medals.shape}")

# 加载东道主数据
df_hosts = pd.read_csv(data_path / "summerOly_hosts.csv")
print(f"东道主数据: {df_hosts.shape}")

# 加载运动员数据
df_athletes = pd.read_csv(data_path / "summerOly_athletes.csv")
print(f"运动员数据: {df_athletes.shape}")

# 加载项目数据（处理编码问题）
try:
    df_programs = pd.read_csv(data_path / "summerOly_programs.csv", encoding='utf-8')
except UnicodeDecodeError:
    df_programs = pd.read_csv(data_path / "summerOly_programs.csv", encoding='latin-1')
print(f"项目数据: {df_programs.shape}")

# ==================== 2. 关键步骤：构建NOC映射表 ====================
print("\n" + "=" * 60)
print("2. 构建NOC映射表（统一国家代码）")
print("=" * 60)

# medals数据中的国家是全名，athletes数据中有NOC代码和Team全名
# 我们需要创建一个完整的映射

# 从athletes数据构建 Team -> NOC 的映射
team_to_noc = df_athletes.groupby('Team')['NOC'].first().to_dict()

# 从athletes数据构建 NOC -> Team 的映射（取最常见的）
noc_to_team = df_athletes.groupby('NOC')['Team'].agg(lambda x: x.value_counts().index[0]).to_dict()

# medals数据中的国家名称列表
medal_countries = df_medals['NOC'].unique()
print(f"奖牌数据中的唯一国家数: {len(medal_countries)}")

# athletes数据中的NOC列表
athlete_nocs = df_athletes['NOC'].unique()
print(f"运动员数据中的唯一NOC数: {len(athlete_nocs)}")

# 创建国家全名到NOC代码的映射
country_to_noc = {}

# 首先使用Team->NOC映射
for team, noc in team_to_noc.items():
    country_to_noc[team] = noc

# 手动补充一些重要的映射
manual_mapping = {
    'United States': 'USA',
    'Great Britain': 'GBR',
    'Soviet Union': 'URS',
    'West Germany': 'FRG',
    'East Germany': 'GDR',
    'China': 'CHN',
    'France': 'FRA',
    'Germany': 'GER',
    'Italy': 'ITA',
    'Australia': 'AUS',
    'Japan': 'JPN',
    'South Korea': 'KOR',
    'North Korea': 'PRK',
    'Netherlands': 'NED',
    'Spain': 'ESP',
    'Canada': 'CAN',
    'Brazil': 'BRA',
    'Russia': 'RUS',
    'Russian Federation': 'RUS',
    'ROC': 'ROC',
    'Unified Team': 'EUN',
    'Greece': 'GRE',
    'Sweden': 'SWE',
    'Hungary': 'HUN',
    'Poland': 'POL',
    'Cuba': 'CUB',
    'Kenya': 'KEN',
    'Jamaica': 'JAM',
    'New Zealand': 'NZL',
    'Romania': 'ROU',
    'Mixed team': 'MIX',  # 特殊处理
}

country_to_noc.update(manual_mapping)

# 为medals数据中的每个国家找到对应的NOC
def get_noc_code(country_name):
    """获取国家的NOC代码"""
    if country_name in country_to_noc:
        return country_to_noc[country_name]
    # 尝试模糊匹配
    country_lower = country_name.lower()
    for team, noc in team_to_noc.items():
        if country_lower in team.lower() or team.lower() in country_lower:
            return noc
    # 如果找不到，返回原名（可能本身就是NOC代码）
    if len(country_name) <= 3 and country_name.isupper():
        return country_name
    return None

# 为medals数据添加标准化的NOC代码
df_medals['NOC_Code'] = df_medals['NOC'].apply(get_noc_code)

# 检查映射结果
unmapped = df_medals[df_medals['NOC_Code'].isna()]['NOC'].unique()
print(f"\n未映射的国家数: {len(unmapped)}")
if len(unmapped) > 0:
    print(f"未映射国家示例: {list(unmapped[:10])}")

# 对于未映射的，尝试直接使用国家名（可能就是NOC）
df_medals['NOC_Code'] = df_medals.apply(
    lambda r: r['NOC_Code'] if pd.notna(r['NOC_Code']) else r['NOC'], 
    axis=1
)

# ==================== 3. 处理项目数据 - 提取每届总项目数 ====================
print("\n" + "=" * 60)
print("3. 处理项目数据")
print("=" * 60)

# 获取年份列
year_columns = []
for col in df_programs.columns:
    col_clean = col.replace('*', '').strip()
    if col_clean.isdigit():
        year_columns.append((col, int(col_clean)))

# 提取Total events行
total_events_row = df_programs[df_programs['Sport'] == 'Total events']
if len(total_events_row) == 0:
    # 尝试其他方式找到总数行
    total_events_row = df_programs[df_programs['Sport'].str.contains('Total', na=False, case=False)]

if len(total_events_row) > 0:
    total_events_row = total_events_row.iloc[0]
    total_events_dict = {}
    for col, year in year_columns:
        try:
            value = total_events_row[col]
            if pd.notna(value):
                # 处理可能的非数字值
                if isinstance(value, (int, float)):
                    total_events_dict[year] = int(value)
                elif str(value).isdigit():
                    total_events_dict[year] = int(value)
        except:
            continue
else:
    # 如果找不到，手动设置已知值
    total_events_dict = {
        1896: 43, 1900: 97, 1904: 95, 1908: 110, 1912: 102,
        1920: 156, 1924: 126, 1928: 109, 1932: 117, 1936: 129,
        1948: 136, 1952: 149, 1956: 151, 1960: 150, 1964: 163,
        1968: 172, 1972: 195, 1976: 198, 1980: 203, 1984: 221,
        1988: 237, 1992: 257, 1996: 271, 2000: 300, 2004: 301,
        2008: 302, 2012: 302, 2016: 306, 2020: 339, 2024: 329
    }

# 补充2028年的项目数
if 2028 not in total_events_dict:
    total_events_dict[2028] = 329  # 预估与2024相近

print(f"项目数数据: {dict(list(sorted(total_events_dict.items()))[-5:])}")

# ==================== 4. 构建投入特征 (从 athletes.csv) ====================
print("\n" + "=" * 60)
print("4. 构建投入特征")
print("=" * 60)

# 4.1 Squad_Size: 每个国家每届奥运会的参赛人数
print("4.1 计算参赛人数 (Squad_Size)...")
squad_size = df_athletes.groupby(['Year', 'NOC'])['Name'].nunique().reset_index()
squad_size.columns = ['Year', 'NOC_Code', 'Squad_Size']
print(f"  参赛人数数据: {squad_size.shape}")

# 4.2 Events_Participated: 每个国家参加了多少个不同的项目
print("4.2 计算参赛项目数 (Events_Participated)...")
events_participated = df_athletes.groupby(['Year', 'NOC'])['Event'].nunique().reset_index()
events_participated.columns = ['Year', 'NOC_Code', 'Events_Participated']

# 4.3 Sports_Participated: 每个国家参加了多少个不同的运动大项
print("4.3 计算参赛大项数 (Sports_Participated)...")
sports_participated = df_athletes.groupby(['Year', 'NOC'])['Sport'].nunique().reset_index()
sports_participated.columns = ['Year', 'NOC_Code', 'Sports_Participated']

# 4.4 Female_Ratio: 女性运动员比例
print("4.4 计算性别比例 (Female_Ratio)...")
gender_stats = df_athletes.groupby(['Year', 'NOC', 'Sex']).size().unstack(fill_value=0).reset_index()
gender_stats.columns.name = None
if 'F' in gender_stats.columns and 'M' in gender_stats.columns:
    gender_stats['Female_Ratio'] = gender_stats['F'] / (gender_stats['F'] + gender_stats['M'])
    gender_stats = gender_stats.rename(columns={'NOC': 'NOC_Code'})[['Year', 'NOC_Code', 'Female_Ratio']]
else:
    gender_stats = pd.DataFrame(columns=['Year', 'NOC_Code', 'Female_Ratio'])

# ==================== 5. 构建基础面板数据 ====================
print("\n" + "=" * 60)
print("5. 构建基础面板数据")
print("=" * 60)

# 从奖牌数据开始
df_panel = df_medals[['NOC', 'NOC_Code', 'Year', 'Gold', 'Silver', 'Bronze', 'Total']].copy()
df_panel = df_panel.rename(columns={'Total': 'Medals'})

# 获取有效年份列表
valid_years = sorted(df_medals['Year'].unique())
print(f"有效奥运年份: {valid_years}")
print(f"基础面板数据: {df_panel.shape}")

# 合并投入特征 (使用NOC_Code进行匹配)
df_panel = df_panel.merge(squad_size, on=['Year', 'NOC_Code'], how='left')
df_panel = df_panel.merge(events_participated, on=['Year', 'NOC_Code'], how='left')
df_panel = df_panel.merge(sports_participated, on=['Year', 'NOC_Code'], how='left')
df_panel = df_panel.merge(gender_stats, on=['Year', 'NOC_Code'], how='left')

# 检查合并效果
print(f"合并后面板数据: {df_panel.shape}")
print(f"Squad_Size非空比例: {df_panel['Squad_Size'].notna().mean():.2%}")

# ==================== 6. 构建东道主特征 ====================
print("\n" + "=" * 60)
print("6. 构建东道主特征")
print("=" * 60)

# 东道主映射 (Year -> NOC全名)
host_mapping_fullname = {
    1896: 'Greece', 1900: 'France', 1904: 'United States', 1908: 'Great Britain',
    1912: 'Sweden', 1920: 'Belgium', 1924: 'France', 1928: 'Netherlands',
    1932: 'United States', 1936: 'Germany', 1948: 'Great Britain', 1952: 'Finland',
    1956: 'Australia', 1960: 'Italy', 1964: 'Japan', 1968: 'Mexico',
    1972: 'West Germany', 1976: 'Canada', 1980: 'Soviet Union', 1984: 'United States',
    1988: 'South Korea', 1992: 'Spain', 1996: 'United States', 2000: 'Australia',
    2004: 'Greece', 2008: 'China', 2012: 'Great Britain', 2016: 'Brazil',
    2020: 'Japan', 2024: 'France', 2028: 'United States', 2032: 'Australia'
}

# 6.1 Is_Host
print("6.1 计算当届东道主 (Is_Host)...")
df_panel['Is_Host'] = df_panel.apply(
    lambda r: 1 if host_mapping_fullname.get(r['Year']) == r['NOC'] else 0, 
    axis=1
)
print(f"  东道主记录数: {df_panel['Is_Host'].sum()}")

# 6.2 Is_Next_Host
print("6.2 计算下届东道主 (Is_Next_Host)...")
next_host_mapping = {}
for i, year in enumerate(valid_years):
    next_years = [y for y in valid_years if y > year]
    if next_years:
        next_year = min(next_years)
        if next_year in host_mapping_fullname:
            next_host_mapping[year] = host_mapping_fullname[next_year]

df_panel['Is_Next_Host'] = df_panel.apply(
    lambda r: 1 if next_host_mapping.get(r['Year']) == r['NOC'] else 0, 
    axis=1
)

# 6.3 Time_Since_Last_Host
print("6.3 计算距离上次举办年数 (Time_Since_Last_Host)...")
host_history = {}
for year, noc in host_mapping_fullname.items():
    if noc not in host_history:
        host_history[noc] = []
    host_history[noc].append(year)

def get_time_since_last_host(row):
    noc = row['NOC']
    year = row['Year']
    if noc not in host_history:
        return 999
    past_hosts = [y for y in host_history[noc] if y < year]
    if not past_hosts:
        return 999
    return year - max(past_hosts)

df_panel['Time_Since_Last_Host'] = df_panel.apply(get_time_since_last_host, axis=1)

# ==================== 7. 构建环境特征 ====================
print("\n" + "=" * 60)
print("7. 构建环境特征")
print("=" * 60)

# 7.1 Total_Events
print("7.1 添加当届总项目数 (Total_Events)...")
df_panel['Total_Events'] = df_panel['Year'].map(total_events_dict)

# 7.2 Region
print("7.2 添加地区分类 (Region)...")

region_mapping = {
    # Ex-USSR
    'Soviet Union': 'Ex-USSR', 'Russia': 'Ex-USSR', 'Unified Team': 'Ex-USSR',
    'Ukraine': 'Ex-USSR', 'Belarus': 'Ex-USSR', 'Kazakhstan': 'Ex-USSR',
    'Uzbekistan': 'Ex-USSR', 'Georgia': 'Ex-USSR', 'Azerbaijan': 'Ex-USSR',
    'Armenia': 'Ex-USSR', 'Moldova': 'Ex-USSR', 'Lithuania': 'Ex-USSR',
    'Latvia': 'Ex-USSR', 'Estonia': 'Ex-USSR', 'ROC': 'Ex-USSR',
    
    # Europe
    'Great Britain': 'Europe', 'France': 'Europe', 'Germany': 'Europe',
    'Italy': 'Europe', 'Spain': 'Europe', 'Netherlands': 'Europe',
    'Belgium': 'Europe', 'Sweden': 'Europe', 'Norway': 'Europe',
    'Denmark': 'Europe', 'Finland': 'Europe', 'Poland': 'Europe',
    'Hungary': 'Europe', 'Austria': 'Europe', 'Switzerland': 'Europe',
    'Czech Republic': 'Europe', 'Czechoslovakia': 'Europe', 'Greece': 'Europe',
    'Romania': 'Europe', 'Bulgaria': 'Europe', 'Serbia': 'Europe',
    'Croatia': 'Europe', 'Slovenia': 'Europe', 'Slovakia': 'Europe',
    'Ireland': 'Europe', 'West Germany': 'Europe', 'East Germany': 'Europe',
    'Portugal': 'Europe', 'Türkiye': 'Europe', 'Turkey': 'Europe',
    
    # Asia
    'China': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia', 'North Korea': 'Asia',
    'India': 'Asia', 'Thailand': 'Asia', 'Indonesia': 'Asia', 'Philippines': 'Asia',
    'Chinese Taipei': 'Asia', 'Hong Kong': 'Asia', 'Iran': 'Asia',
    'Pakistan': 'Asia', 'Mongolia': 'Asia', 'Vietnam': 'Asia',
    
    # Americas
    'United States': 'Americas', 'Canada': 'Americas', 'Mexico': 'Americas',
    'Brazil': 'Americas', 'Argentina': 'Americas', 'Cuba': 'Americas',
    'Jamaica': 'Americas', 'Colombia': 'Americas', 'Venezuela': 'Americas',
    'Chile': 'Americas', 'Peru': 'Americas', 'Ecuador': 'Americas',
    'Puerto Rico': 'Americas', 'Trinidad and Tobago': 'Americas',
    'Bahamas': 'Americas', 'Dominican Republic': 'Americas',
    
    # Oceania
    'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Fiji': 'Oceania',
    
    # Africa
    'South Africa': 'Africa', 'Kenya': 'Africa', 'Ethiopia': 'Africa',
    'Nigeria': 'Africa', 'Egypt': 'Africa', 'Morocco': 'Africa',
    'Algeria': 'Africa', 'Tunisia': 'Africa', 'Ghana': 'Africa',
    'Cameroon': 'Africa', 'Zimbabwe': 'Africa', 'Uganda': 'Africa',
}

df_panel['Region'] = df_panel['NOC'].map(region_mapping).fillna('Other')
print(f"地区分布:\n{df_panel['Region'].value_counts()}")

# 7.3 Is_Post_1992
df_panel['Is_Post_1992'] = (df_panel['Year'] > 1992).astype(int)

# ==================== 8. 构建惯性特征 ====================
print("\n" + "=" * 60)
print("8. 构建惯性特征 (Momentum Features)")
print("=" * 60)

# 按NOC和Year排序
df_panel = df_panel.sort_values(['NOC', 'Year']).reset_index(drop=True)

# 年份映射
def get_prev_year(year, n=1):
    """获取前n届的年份"""
    try:
        idx = valid_years.index(year)
        if idx >= n:
            return valid_years[idx - n]
    except ValueError:
        pass
    return None

# 创建查找字典
medals_lookup = df_panel.set_index(['NOC', 'Year'])['Medals'].to_dict()
squad_lookup = df_panel.set_index(['NOC', 'Year'])['Squad_Size'].to_dict()

def get_lag_value(noc, year, lookup_dict, n=1):
    prev_year = get_prev_year(year, n)
    if prev_year is None:
        return np.nan
    return lookup_dict.get((noc, prev_year), np.nan)

# 8.1 Lag Features
print("8.1 计算滞后特征...")
df_panel['Lag1_Medals'] = df_panel.apply(lambda r: get_lag_value(r['NOC'], r['Year'], medals_lookup, 1), axis=1)
df_panel['Lag2_Medals'] = df_panel.apply(lambda r: get_lag_value(r['NOC'], r['Year'], medals_lookup, 2), axis=1)
df_panel['Lag3_Medals'] = df_panel.apply(lambda r: get_lag_value(r['NOC'], r['Year'], medals_lookup, 3), axis=1)

# 8.2 Weighted Average
print("8.2 计算加权平均 (Weighted_Avg_3)...")
def weighted_avg_3(row):
    lag1 = row['Lag1_Medals']
    lag2 = row['Lag2_Medals']
    lag3 = row['Lag3_Medals']
    
    values, weights = [], []
    if pd.notna(lag1):
        values.append(lag1)
        weights.append(0.5)
    if pd.notna(lag2):
        values.append(lag2)
        weights.append(0.3)
    if pd.notna(lag3):
        values.append(lag3)
        weights.append(0.2)
    
    if not values:
        return np.nan
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

df_panel['Weighted_Avg_3'] = df_panel.apply(weighted_avg_3, axis=1)

# 8.3 EWMA
print("8.3 计算指数加权移动平均 (EWMA)...")
alpha = 0.4
ewma_results = {}
for noc, group in df_panel.groupby('NOC'):
    group = group.sort_values('Year')
    prev_score = None
    for _, row in group.iterrows():
        medals = row['Medals']
        if prev_score is None:
            score = medals
        else:
            score = alpha * medals + (1 - alpha) * prev_score
        ewma_results[(noc, row['Year'])] = score
        prev_score = score

df_panel['EWMA_Score'] = df_panel.apply(lambda r: ewma_results.get((r['NOC'], r['Year']), np.nan), axis=1)

# Lag EWMA
df_panel['Lag1_EWMA'] = df_panel.apply(
    lambda r: ewma_results.get((r['NOC'], get_prev_year(r['Year'], 1)), np.nan) if get_prev_year(r['Year'], 1) else np.nan, 
    axis=1
)

# 8.4 Momentum
print("8.4 计算趋势指标 (Momentum)...")
df_panel['Momentum'] = df_panel['Medals'] - df_panel['Lag1_Medals']

# ==================== 9. 构建效率特征 ====================
print("\n" + "=" * 60)
print("9. 构建效率特征")
print("=" * 60)

# 9.1 Efficiency
print("9.1 计算奖牌效率 (Efficiency)...")
df_panel['Efficiency'] = df_panel['Medals'] / df_panel['Squad_Size'].replace(0, np.nan)

# 9.2 Lag Efficiency
efficiency_lookup = df_panel.set_index(['NOC', 'Year'])['Efficiency'].to_dict()
df_panel['Lag1_Efficiency'] = df_panel.apply(
    lambda r: get_lag_value(r['NOC'], r['Year'], efficiency_lookup, 1), 
    axis=1
)

# 9.3 Lag Squad Size
df_panel['Lag1_Squad_Size'] = df_panel.apply(
    lambda r: get_lag_value(r['NOC'], r['Year'], squad_lookup, 1), 
    axis=1
)

# 9.4 Event Coverage
print("9.2 计算项目覆盖率 (Event_Coverage)...")
df_panel['Event_Coverage'] = df_panel['Events_Participated'] / df_panel['Total_Events'].replace(0, np.nan)

# 9.5 Gold Ratio
print("9.3 计算金牌占比 (Gold_Ratio)...")
df_panel['Gold_Ratio'] = df_panel['Gold'] / df_panel['Medals'].replace(0, np.nan)

# ==================== 10. 特殊历史情况处理 ====================
print("\n" + "=" * 60)
print("10. 特殊历史情况处理")
print("=" * 60)

# 10.1 抵制年份标记
print("10.1 标记抵制年份...")
boycott_1980 = ['United States', 'West Germany', 'Japan', 'Canada']
boycott_1984 = ['Soviet Union', 'East Germany', 'Cuba', 'Bulgaria', 'Czechoslovakia']

def is_boycott_year(row):
    year = row['Year']
    noc = row['NOC']
    if year == 1980 and noc in boycott_1980:
        return 1
    if year == 1984 and noc in boycott_1984:
        return 1
    return 0

df_panel['Is_Boycott_Year'] = df_panel.apply(is_boycott_year, axis=1)

# 10.2 历史数据权重
print("10.2 添加历史数据权重...")
df_panel['Historical_Weight'] = np.where(df_panel['Year'] < 1992, 0.7, 1.0)

# ==================== 11. 数据清洗 ====================
print("\n" + "=" * 60)
print("11. 数据质量检查和清洗")
print("=" * 60)

print(f"\n数据集大小: {df_panel.shape}")
print(f"\n缺失值统计:")
null_counts = df_panel.isnull().sum()
print(null_counts[null_counts > 0])

# 填充数值型缺失值
fill_cols = ['Squad_Size', 'Events_Participated', 'Sports_Participated', 
             'Female_Ratio', 'Total_Events', 'Event_Coverage']
for col in fill_cols:
    if col in df_panel.columns and df_panel[col].isna().any():
        median_val = df_panel[col].median()
        df_panel[col] = df_panel[col].fillna(median_val)

# ==================== 12. 创建最终训练数据集 ====================
print("\n" + "=" * 60)
print("12. 创建最终训练数据集")
print("=" * 60)

feature_cols = [
    # 标识列
    'NOC', 'NOC_Code', 'Year',
    
    # 目标变量
    'Medals', 'Gold', 'Silver', 'Bronze',
    
    # 惯性特征
    'Lag1_Medals', 'Lag2_Medals', 'Lag3_Medals',
    'Weighted_Avg_3', 'EWMA_Score', 'Lag1_EWMA', 'Momentum',
    
    # 投入特征
    'Squad_Size', 'Lag1_Squad_Size',
    'Events_Participated', 'Sports_Participated', 'Female_Ratio',
    
    # 效率特征
    'Efficiency', 'Lag1_Efficiency', 'Event_Coverage', 'Gold_Ratio',
    
    # 东道主特征
    'Is_Host', 'Is_Next_Host', 'Time_Since_Last_Host',
    
    # 环境特征
    'Total_Events', 'Region', 'Is_Post_1992',
    
    # 特殊标记
    'Is_Boycott_Year', 'Historical_Weight'
]

df_train = df_panel[feature_cols].copy()
df_train = df_train.rename(columns={'Medals': 'Target'})

print(f"最终训练数据集: {df_train.shape}")
print(f"\n特征列表 ({len(df_train.columns)} 列):")
for i, col in enumerate(df_train.columns, 1):
    print(f"  {i:2d}. {col}")

# ==================== 13. 保存数据 ====================
print("\n" + "=" * 60)
print("13. 保存数据")
print("=" * 60)

# 完整面板数据
df_train.to_csv("df_train_panel.csv", index=False)
print(f"已保存: df_train_panel.csv ({df_train.shape[0]} rows)")

# 有效滞后特征的数据
df_train_valid = df_train.dropna(subset=['Lag1_Medals'])
df_train_valid.to_csv("df_train_valid.csv", index=False)
print(f"已保存: df_train_valid.csv ({df_train_valid.shape[0]} rows)")

# 现代数据 (1984年后)
df_train_modern = df_train[df_train['Year'] >= 1984].copy()
df_train_modern.to_csv("df_train_modern.csv", index=False)
print(f"已保存: df_train_modern.csv ({df_train_modern.shape[0]} rows)")

# ==================== 14. 数据验证 ====================
print("\n" + "=" * 60)
print("14. 数据验证")
print("=" * 60)

# 美国数据
print("\n🇺🇸 美国 (United States) 近5届数据:")
usa_data = df_train[df_train['NOC'] == 'United States'].sort_values('Year').tail(5)
print(usa_data[['NOC', 'Year', 'Target', 'Lag1_Medals', 'Squad_Size', 'Is_Host', 'EWMA_Score']].to_string(index=False))

# 中国数据
print("\n🇨🇳 中国 (China) 近5届数据:")
china_data = df_train[df_train['NOC'] == 'China'].sort_values('Year').tail(5)
print(china_data[['NOC', 'Year', 'Target', 'Lag1_Medals', 'Squad_Size', 'Is_Host', 'EWMA_Score']].to_string(index=False))

# 东道主效应
print("\n🏠 东道主效应分析 (近10届):")
host_data = df_train[df_train['Is_Host'] == 1][['NOC', 'Year', 'Target', 'Lag1_Medals']].copy()
host_data['Host_Boost'] = host_data['Target'] - host_data['Lag1_Medals']
host_data['Boost_Pct'] = (host_data['Host_Boost'] / host_data['Lag1_Medals'] * 100).round(1)
print(host_data.sort_values('Year').tail(10).to_string(index=False))

# 统计摘要
print("\n📊 关键特征统计:")
key_features = ['Target', 'Squad_Size', 'EWMA_Score', 'Efficiency', 'Event_Coverage']
print(df_train[key_features].describe().round(2).to_string())

print("\n" + "=" * 60)
print("✅ 数据工程完成!")
print("=" * 60)
