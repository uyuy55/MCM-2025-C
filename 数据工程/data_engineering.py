"""
数据工程阶段：构建面板数据 (Panel Data)
整合 medal_counts.csv, hosts.csv, athletes.csv 和 programs.csv
构建训练数据集 df_train，每一行是一个"国家-年份"
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

# ==================== 2. 数据预处理 ====================
print("\n" + "=" * 60)
print("2. 数据预处理")
print("=" * 60)

# 2.1 处理东道主数据
print("\n2.1 处理东道主数据...")

# 清理东道主数据，移除取消的奥运会
df_hosts_clean = df_hosts[~df_hosts['Host'].str.contains('Cancelled', na=False)].copy()
df_hosts_clean['Host'] = df_hosts_clean['Host'].str.strip()

# 从Host字段提取国家
def extract_host_country(host_str):
    """从Host字符串提取国家名称"""
    if pd.isna(host_str):
        return None
    parts = host_str.split(',')
    if len(parts) >= 2:
        return parts[-1].strip()
    return host_str.strip()

df_hosts_clean['Host_Country'] = df_hosts_clean['Host'].apply(extract_host_country)

# 国家名到NOC的映射（手动创建关键映射）
country_to_noc = {
    'Greece': 'GRE',
    'France': 'FRA',
    'United States': 'USA',
    'United Kingdom': 'GBR',
    'Sweden': 'SWE',
    'Belgium': 'BEL',
    'Netherlands': 'NED',
    'Germany': 'GER',
    'Finland': 'FIN',
    'Australia': 'AUS',
    'Italy': 'ITA',
    'Japan': 'JPN',
    'Mexico': 'MEX',
    'West Germany': 'GER',  # 西德用德国的NOC
    'Canada': 'CAN',
    'Soviet Union': 'URS',
    'South Korea': 'KOR',
    'Spain': 'ESP',
    'China': 'CHN',
    'Brazil': 'BRA'
}

# 获取所有唯一的NOC（从medals数据中获取）
all_nocs = df_medals['NOC'].unique()

# 更完善的国家名到NOC映射
# 从athletes数据构建Team到NOC的映射
team_noc_mapping = df_athletes.groupby('Team')['NOC'].first().to_dict()

def get_host_noc(country_name):
    """获取东道主国家的NOC代码"""
    if pd.isna(country_name):
        return None
    # 首先检查预定义映射
    if country_name in country_to_noc:
        return country_to_noc[country_name]
    # 然后检查team映射
    if country_name in team_noc_mapping:
        return team_noc_mapping[country_name]
    # 最后尝试在NOC列表中查找
    for noc in all_nocs:
        if country_name.lower() in noc.lower():
            return noc
    return None

df_hosts_clean['Host_NOC'] = df_hosts_clean['Host_Country'].apply(get_host_noc)

print("东道主数据示例:")
print(df_hosts_clean.head(10))

# 2.2 处理项目数据 - 提取每届总项目数
print("\n2.2 处理项目数据...")

# 获取年份列（从1896到2024）
year_columns = [col for col in df_programs.columns if col.isdigit() or (col.endswith('*') and col[:-1].isdigit())]

# 提取Total events行
total_events_row = df_programs[df_programs['Sport'] == 'Total events'].iloc[0]

# 构建年份-项目数字典
total_events_dict = {}
for col in df_programs.columns[4:]:  # 跳过前4列（Sport, Discipline, Code, Sports Governing Body）
    try:
        # 处理年份列名（如 "1906*"）
        year_str = col.replace('*', '')
        if year_str.isdigit():
            year = int(year_str)
            value = total_events_row[col]
            if pd.notna(value) and str(value).isdigit():
                total_events_dict[year] = int(value)
    except:
        continue

# 补充2028年的项目数（预估，与2024相近）
if 2028 not in total_events_dict:
    total_events_dict[2028] = total_events_dict.get(2024, 329)

print(f"项目数数据示例: {dict(list(total_events_dict.items())[-5:])}")

# 2.3 处理奖牌数据
print("\n2.3 处理奖牌数据...")

# 检查并清理NOC字段
print(f"唯一国家数: {df_medals['NOC'].nunique()}")
print(f"年份范围: {df_medals['Year'].min()} - {df_medals['Year'].max()}")

# 处理历史国家代码变化（重要！）
# 苏联/独联体/俄罗斯等历史变迁
historical_noc_mapping = {
    'Soviet Union': 'URS',
    'Unified Team': 'EUN',  # 1992独联体
    'Russian Olympic Committee': 'ROC',  # 2020/2024
    'Olympic Athletes from Russia': 'OAR',  # 2018冬奥
}

# ==================== 3. 构建投入特征 (从 athletes.csv) ====================
print("\n" + "=" * 60)
print("3. 构建投入特征 (从 athletes.csv)")
print("=" * 60)

# 3.1 Squad_Size: 每个国家每届奥运会的参赛人数
print("\n3.1 计算参赛人数 (Squad_Size)...")
squad_size = df_athletes.groupby(['Year', 'NOC'])['Name'].nunique().reset_index()
squad_size.columns = ['Year', 'NOC', 'Squad_Size']
print(f"参赛人数数据: {squad_size.shape}")

# 3.2 Events_Participated: 每个国家参加了多少个不同的项目
print("\n3.2 计算参赛项目数 (Events_Participated)...")
events_participated = df_athletes.groupby(['Year', 'NOC'])['Event'].nunique().reset_index()
events_participated.columns = ['Year', 'NOC', 'Events_Participated']
print(f"参赛项目数据: {events_participated.shape}")

# 3.3 Sports_Participated: 每个国家参加了多少个不同的运动大项
print("\n3.3 计算参赛大项数 (Sports_Participated)...")
sports_participated = df_athletes.groupby(['Year', 'NOC'])['Sport'].nunique().reset_index()
sports_participated.columns = ['Year', 'NOC', 'Sports_Participated']

# 3.4 Gender_Ratio: 男女运动员比例（可能反映国家体育发展水平）
print("\n3.4 计算性别比例 (Female_Ratio)...")
gender_stats = df_athletes.groupby(['Year', 'NOC', 'Sex'])['Name'].nunique().unstack(fill_value=0).reset_index()
gender_stats.columns.name = None
if 'F' in gender_stats.columns and 'M' in gender_stats.columns:
    gender_stats['Female_Ratio'] = gender_stats['F'] / (gender_stats['F'] + gender_stats['M'])
else:
    gender_stats['Female_Ratio'] = 0.0
gender_stats = gender_stats[['Year', 'NOC', 'Female_Ratio']]

# ==================== 4. 构建基础面板数据 ====================
print("\n" + "=" * 60)
print("4. 构建基础面板数据")
print("=" * 60)

# 从奖牌数据开始构建
df_panel = df_medals[['NOC', 'Year', 'Gold', 'Silver', 'Bronze', 'Total']].copy()
df_panel = df_panel.rename(columns={'Total': 'Medals'})

# 获取有效年份列表（排除取消的）
valid_years = sorted(df_medals['Year'].unique())
print(f"有效奥运年份: {valid_years}")

# 合并投入特征
df_panel = df_panel.merge(squad_size, on=['Year', 'NOC'], how='left')
df_panel = df_panel.merge(events_participated, on=['Year', 'NOC'], how='left')
df_panel = df_panel.merge(sports_participated, on=['Year', 'NOC'], how='left')
df_panel = df_panel.merge(gender_stats, on=['Year', 'NOC'], how='left')

print(f"合并后面板数据: {df_panel.shape}")

# ==================== 5. 构建东道主特征 ====================
print("\n" + "=" * 60)
print("5. 构建东道主特征")
print("=" * 60)

# 5.1 Is_Host: 是否为当届东道主
print("\n5.1 计算当届东道主 (Is_Host)...")
host_years = df_hosts_clean[['Year', 'Host_NOC']].dropna()
host_years_dict = dict(zip(host_years['Year'], host_years['Host_NOC']))

# 也需要处理NOC名称匹配
# 例如 "United States" 在 medals 数据中可能是 "United States" 而不是 "USA"
# 我们需要创建一个完整的映射

# 从medals数据获取国家全名到简称的映射
noc_full_names = df_medals['NOC'].unique()
print(f"奖牌数据中的国家示例: {list(noc_full_names[:10])}")

# 创建Is_Host标志
def check_is_host(row, host_dict):
    year = row['Year']
    noc = row['NOC']
    if year not in host_dict:
        return 0
    host_noc = host_dict[year]
    # 检查直接匹配
    if noc == host_noc:
        return 1
    # 检查国家名匹配
    if host_noc in country_to_noc:
        if noc == country_to_noc[host_noc]:
            return 1
    # 检查反向匹配
    for country, code in country_to_noc.items():
        if code == host_noc and noc == country:
            return 1
        if code == noc and country == host_noc:
            return 1
    return 0

# 重新构建更准确的host映射
host_mapping = {}
for _, row in df_hosts_clean.iterrows():
    year = row['Year']
    host_country = row['Host_Country']
    # 尝试匹配medals数据中的NOC
    for noc in noc_full_names:
        if host_country and (host_country in noc or noc in host_country):
            host_mapping[year] = noc
            break
    # 使用预定义映射
    if year not in host_mapping and host_country in country_to_noc:
        # 检查medals数据中是否有对应的NOC
        target_noc = country_to_noc[host_country]
        for noc in noc_full_names:
            if target_noc in noc or noc in target_noc:
                host_mapping[year] = noc
                break

# 手动修正一些映射
manual_host_mapping = {
    1896: 'Greece',
    1900: 'France',
    1904: 'United States',
    1908: 'Great Britain',
    1912: 'Sweden',
    1920: 'Belgium',
    1924: 'France',
    1928: 'Netherlands',
    1932: 'United States',
    1936: 'Germany',
    1948: 'Great Britain',
    1952: 'Finland',
    1956: 'Australia',
    1960: 'Italy',
    1964: 'Japan',
    1968: 'Mexico',
    1972: 'West Germany',  # 需要特殊处理
    1976: 'Canada',
    1980: 'Soviet Union',
    1984: 'United States',
    1988: 'South Korea',
    1992: 'Spain',
    1996: 'United States',
    2000: 'Australia',
    2004: 'Greece',
    2008: 'China',
    2012: 'Great Britain',
    2016: 'Brazil',
    2020: 'Japan',
    2024: 'France',
    2028: 'United States'
}

# 更新为medals数据中的实际国家名
# 首先检查medals数据中的实际国家名
for year, country in manual_host_mapping.items():
    if country not in noc_full_names:
        # 尝试找到匹配的
        for noc in noc_full_names:
            if country.lower() in noc.lower() or noc.lower() in country.lower():
                manual_host_mapping[year] = noc
                break

print("东道主映射:")
for year in sorted(manual_host_mapping.keys())[-10:]:
    print(f"  {year}: {manual_host_mapping[year]}")

df_panel['Is_Host'] = df_panel.apply(
    lambda row: 1 if manual_host_mapping.get(row['Year']) == row['NOC'] else 0, 
    axis=1
)

# 5.2 Is_Next_Host: 是否为下届东道主
print("\n5.2 计算下届东道主 (Is_Next_Host)...")
# 构建下届东道主映射
next_host_mapping = {}
for year in valid_years:
    # 找到下一届年份
    future_years = [y for y in valid_years if y > year]
    if future_years:
        next_year = min(future_years)
        if next_year in manual_host_mapping:
            next_host_mapping[year] = manual_host_mapping[next_year]

df_panel['Is_Next_Host'] = df_panel.apply(
    lambda row: 1 if next_host_mapping.get(row['Year']) == row['NOC'] else 0, 
    axis=1
)

# 5.3 Time_Since_Last_Host: 距离上次举办的年数
print("\n5.3 计算距离上次举办年数 (Time_Since_Last_Host)...")

# 首先构建每个国家的举办历史
host_history = {}
for year, noc in manual_host_mapping.items():
    if noc not in host_history:
        host_history[noc] = []
    host_history[noc].append(year)

def get_time_since_last_host(row):
    noc = row['NOC']
    year = row['Year']
    if noc not in host_history:
        return 999  # 从未举办过
    past_hosts = [y for y in host_history[noc] if y < year]
    if not past_hosts:
        return 999
    return year - max(past_hosts)

df_panel['Time_Since_Last_Host'] = df_panel.apply(get_time_since_last_host, axis=1)

# ==================== 6. 构建环境特征 ====================
print("\n" + "=" * 60)
print("6. 构建环境特征")
print("=" * 60)

# 6.1 Total_Events: 当届总项目数
print("\n6.1 添加当届总项目数 (Total_Events)...")
df_panel['Total_Events'] = df_panel['Year'].map(total_events_dict)

# 6.2 Region: 地区分类（处理苏联/独联体效应）
print("\n6.2 添加地区分类 (Region)...")

# 定义地区映射
ex_ussr_countries = [
    'Soviet Union', 'Russia', 'Russian Federation', 'ROC', 'OAR',
    'Ukraine', 'Belarus', 'Kazakhstan', 'Uzbekistan', 'Georgia',
    'Azerbaijan', 'Armenia', 'Moldova', 'Turkmenistan', 'Tajikistan',
    'Kyrgyzstan', 'Lithuania', 'Latvia', 'Estonia', 'Unified Team'
]

europe_countries = [
    'Great Britain', 'France', 'Germany', 'Italy', 'Spain', 'Netherlands',
    'Belgium', 'Sweden', 'Norway', 'Denmark', 'Finland', 'Poland',
    'Czech Republic', 'Czechoslovakia', 'Hungary', 'Austria', 'Switzerland',
    'Portugal', 'Greece', 'Romania', 'Bulgaria', 'Serbia', 'Croatia',
    'Slovenia', 'Slovakia', 'Ireland', 'West Germany', 'East Germany'
]

asia_countries = [
    'China', 'Japan', 'South Korea', 'North Korea', 'India', 'Thailand',
    'Indonesia', 'Philippines', 'Vietnam', 'Malaysia', 'Singapore',
    'Chinese Taipei', 'Hong Kong', 'Iran', 'Iraq', 'Saudi Arabia',
    'Pakistan', 'Bangladesh', 'Sri Lanka', 'Mongolia'
]

americas_countries = [
    'United States', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Cuba',
    'Jamaica', 'Colombia', 'Venezuela', 'Chile', 'Peru', 'Ecuador',
    'Puerto Rico', 'Trinidad and Tobago', 'Bahamas', 'Dominican Republic'
]

oceania_countries = [
    'Australia', 'New Zealand', 'Fiji', 'Papua New Guinea'
]

africa_countries = [
    'South Africa', 'Kenya', 'Ethiopia', 'Nigeria', 'Egypt', 'Morocco',
    'Algeria', 'Tunisia', 'Ghana', 'Cameroon', 'Ivory Coast', 'Zimbabwe'
]

def get_region(noc):
    noc_lower = noc.lower() if isinstance(noc, str) else ''
    
    # Ex-USSR
    for country in ex_ussr_countries:
        if country.lower() in noc_lower or noc_lower in country.lower():
            return 'Ex-USSR'
    
    # Europe
    for country in europe_countries:
        if country.lower() in noc_lower or noc_lower in country.lower():
            return 'Europe'
    
    # Asia
    for country in asia_countries:
        if country.lower() in noc_lower or noc_lower in country.lower():
            return 'Asia'
    
    # Americas
    for country in americas_countries:
        if country.lower() in noc_lower or noc_lower in country.lower():
            return 'Americas'
    
    # Oceania
    for country in oceania_countries:
        if country.lower() in noc_lower or noc_lower in country.lower():
            return 'Oceania'
    
    # Africa
    for country in africa_countries:
        if country.lower() in noc_lower or noc_lower in country.lower():
            return 'Africa'
    
    return 'Other'

df_panel['Region'] = df_panel['NOC'].apply(get_region)
print(f"地区分布:\n{df_panel['Region'].value_counts()}")

# 6.3 Is_Post_1992: 是否为1992年后的数据（苏联解体后）
df_panel['Is_Post_1992'] = (df_panel['Year'] > 1992).astype(int)

# ==================== 7. 构建惯性特征 (Momentum Features) ====================
print("\n" + "=" * 60)
print("7. 构建惯性特征 (Momentum Features)")
print("=" * 60)

# 首先需要按NOC和Year排序
df_panel = df_panel.sort_values(['NOC', 'Year']).reset_index(drop=True)

# 构建年份索引映射
year_to_prev = {}
for i, year in enumerate(valid_years):
    if i > 0:
        year_to_prev[year] = valid_years[i-1]

year_to_prev2 = {}
for i, year in enumerate(valid_years):
    if i > 1:
        year_to_prev2[year] = valid_years[i-2]

year_to_prev3 = {}
for i, year in enumerate(valid_years):
    if i > 2:
        year_to_prev3[year] = valid_years[i-3]

# 创建历史数据查找字典
medals_lookup = df_panel.set_index(['NOC', 'Year'])['Medals'].to_dict()
squad_lookup = df_panel.set_index(['NOC', 'Year'])['Squad_Size'].to_dict()

# 7.1 Lag Features: 滞后奖牌数
print("\n7.1 计算滞后特征 (Lag1, Lag2, Lag3)...")

def get_lag_value(row, lookup_dict, year_map):
    noc = row['NOC']
    year = row['Year']
    if year not in year_map:
        return np.nan
    prev_year = year_map[year]
    return lookup_dict.get((noc, prev_year), np.nan)

df_panel['Lag1_Medals'] = df_panel.apply(lambda r: get_lag_value(r, medals_lookup, year_to_prev), axis=1)
df_panel['Lag2_Medals'] = df_panel.apply(lambda r: get_lag_value(r, medals_lookup, year_to_prev2), axis=1)
df_panel['Lag3_Medals'] = df_panel.apply(lambda r: get_lag_value(r, medals_lookup, year_to_prev3), axis=1)

# 7.2 Weighted Average: 加权平均（越近权重越大）
print("\n7.2 计算加权平均 (Weighted_Avg_3)...")

def weighted_avg_3(row):
    """过去三届加权平均，权重为 0.5, 0.3, 0.2"""
    lag1 = row['Lag1_Medals']
    lag2 = row['Lag2_Medals']
    lag3 = row['Lag3_Medals']
    
    values = []
    weights = []
    
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
    
    # 重新归一化权重
    total_weight = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total_weight

df_panel['Weighted_Avg_3'] = df_panel.apply(weighted_avg_3, axis=1)

# 7.3 EWMA: 指数加权移动平均
print("\n7.3 计算指数加权移动平均 (EWMA)...")

def calculate_ewma(group, alpha=0.4):
    """计算EWMA分数"""
    group = group.sort_values('Year')
    ewma_scores = []
    prev_score = None
    
    for _, row in group.iterrows():
        medals = row['Medals']
        if prev_score is None:
            ewma_scores.append(medals)
            prev_score = medals
        else:
            score = alpha * medals + (1 - alpha) * prev_score
            ewma_scores.append(score)
            prev_score = score
    
    return ewma_scores

# 按国家计算EWMA
ewma_results = []
for noc, group in df_panel.groupby('NOC'):
    group = group.sort_values('Year')
    ewma_values = calculate_ewma(group)
    for idx, ewma in zip(group.index, ewma_values):
        ewma_results.append((idx, ewma))

ewma_df = pd.DataFrame(ewma_results, columns=['index', 'EWMA_Score'])
ewma_df = ewma_df.set_index('index')
df_panel['EWMA_Score'] = ewma_df['EWMA_Score']

# 7.4 Momentum: 趋势指标（本届 vs 上届）
print("\n7.4 计算趋势指标 (Momentum)...")
df_panel['Momentum'] = df_panel['Medals'] - df_panel['Lag1_Medals']

# 7.5 Lag EWMA (用于预测)
# 创建滞后的EWMA分数
ewma_lookup = df_panel.set_index(['NOC', 'Year'])['EWMA_Score'].to_dict()
df_panel['Lag1_EWMA'] = df_panel.apply(lambda r: get_lag_value(r, ewma_lookup, year_to_prev), axis=1)

# ==================== 8. 构建效率特征 ====================
print("\n" + "=" * 60)
print("8. 构建效率特征")
print("=" * 60)

# 8.1 Efficiency: 奖牌效率 (Medals / Squad_Size)
print("\n8.1 计算奖牌效率 (Efficiency)...")
df_panel['Efficiency'] = df_panel['Medals'] / df_panel['Squad_Size'].replace(0, np.nan)

# 8.2 Lag Efficiency
efficiency_lookup = df_panel.set_index(['NOC', 'Year'])['Efficiency'].to_dict()
df_panel['Lag1_Efficiency'] = df_panel.apply(lambda r: get_lag_value(r, efficiency_lookup, year_to_prev), axis=1)

# 8.3 Event Coverage: 项目覆盖率
print("\n8.2 计算项目覆盖率 (Event_Coverage)...")
df_panel['Event_Coverage'] = df_panel['Events_Participated'] / df_panel['Total_Events'].replace(0, np.nan)

# 8.4 Gold Ratio: 金牌占比
print("\n8.3 计算金牌占比 (Gold_Ratio)...")
df_panel['Gold_Ratio'] = df_panel['Gold'] / df_panel['Medals'].replace(0, np.nan)

# 8.5 Lag Squad Size
df_panel['Lag1_Squad_Size'] = df_panel.apply(lambda r: get_lag_value(r, squad_lookup, year_to_prev), axis=1)

# ==================== 9. 处理特殊历史情况 ====================
print("\n" + "=" * 60)
print("9. 处理特殊历史情况")
print("=" * 60)

# 9.1 标记抵制年份
print("\n9.1 标记抵制年份...")
# 1980年莫斯科奥运会被西方国家抵制
# 1984年洛杉矶奥运会被东方阵营抵制
boycott_1980 = ['United States', 'West Germany', 'Japan', 'Canada']
boycott_1984 = ['Soviet Union', 'East Germany', 'Cuba', 'Bulgaria', 'Czechoslovakia', 'Hungary', 'Poland']

def is_boycott_year(row):
    year = row['Year']
    noc = row['NOC']
    if year == 1980:
        for country in boycott_1980:
            if country.lower() in noc.lower() or noc.lower() in country.lower():
                return 1
    if year == 1984:
        for country in boycott_1984:
            if country.lower() in noc.lower() or noc.lower() in country.lower():
                return 1
    return 0

df_panel['Is_Boycott_Year'] = df_panel.apply(is_boycott_year, axis=1)

# 9.2 对1992年前的数据进行降权标记
print("\n9.2 添加历史数据权重 (Historical_Weight)...")
df_panel['Historical_Weight'] = np.where(df_panel['Year'] < 1992, 0.7, 1.0)

# ==================== 10. 数据质量检查和清洗 ====================
print("\n" + "=" * 60)
print("10. 数据质量检查和清洗")
print("=" * 60)

print(f"\n数据集大小: {df_panel.shape}")
print(f"\n缺失值统计:")
print(df_panel.isnull().sum())

# 填充缺失值
print("\n填充缺失值...")

# 对于数值型特征，使用0或中位数填充
numeric_cols = ['Squad_Size', 'Events_Participated', 'Sports_Participated', 
                'Female_Ratio', 'Total_Events', 'Event_Coverage']
for col in numeric_cols:
    if col in df_panel.columns:
        df_panel[col] = df_panel[col].fillna(df_panel[col].median())

# Lag特征保留NA（用于模型训练时筛选）

# ==================== 11. 创建最终训练数据集 ====================
print("\n" + "=" * 60)
print("11. 创建最终训练数据集")
print("=" * 60)

# 选择最终特征
feature_cols = [
    # 标识列
    'NOC', 'Year',
    
    # 目标变量
    'Medals', 'Gold', 'Silver', 'Bronze',
    
    # 惯性特征
    'Lag1_Medals', 'Lag2_Medals', 'Lag3_Medals',
    'Weighted_Avg_3', 'EWMA_Score', 'Lag1_EWMA', 'Momentum',
    
    # 投入特征
    'Squad_Size', 'Lag1_Squad_Size',
    'Events_Participated', 'Sports_Participated',
    'Female_Ratio',
    
    # 效率特征
    'Efficiency', 'Lag1_Efficiency',
    'Event_Coverage', 'Gold_Ratio',
    
    # 东道主特征
    'Is_Host', 'Is_Next_Host', 'Time_Since_Last_Host',
    
    # 环境特征
    'Total_Events', 'Region', 'Is_Post_1992',
    
    # 特殊标记
    'Is_Boycott_Year', 'Historical_Weight'
]

df_train = df_panel[feature_cols].copy()

# 重命名目标变量
df_train = df_train.rename(columns={'Medals': 'Target'})

print(f"\n最终训练数据集: {df_train.shape}")
print(f"\n特征列表:")
for col in df_train.columns:
    print(f"  - {col}")

print(f"\n数据预览:")
print(df_train.head(10))

print(f"\n数据统计:")
print(df_train.describe())

# ==================== 12. 保存数据 ====================
print("\n" + "=" * 60)
print("12. 保存数据")
print("=" * 60)

# 保存完整训练数据
df_train.to_csv("df_train_panel.csv", index=False)
print("已保存: df_train_panel.csv")

# 保存只有有效Lag特征的数据（用于实际训练）
df_train_valid = df_train.dropna(subset=['Lag1_Medals'])
df_train_valid.to_csv("df_train_valid.csv", index=False)
print(f"已保存: df_train_valid.csv ({df_train_valid.shape[0]} rows)")

# 保存最近10届的数据（1984-2024，更相关的现代数据）
df_train_modern = df_train[df_train['Year'] >= 1984].copy()
df_train_modern.to_csv("df_train_modern.csv", index=False)
print(f"已保存: df_train_modern.csv ({df_train_modern.shape[0]} rows)")

print("\n" + "=" * 60)
print("数据工程完成!")
print("=" * 60)

# ==================== 13. 数据验证 ====================
print("\n" + "=" * 60)
print("13. 数据验证示例")
print("=" * 60)

# 检查美国的数据
print("\n美国 (United States) 近5届数据:")
usa_data = df_train[df_train['NOC'] == 'United States'].sort_values('Year').tail(5)
print(usa_data[['NOC', 'Year', 'Target', 'Lag1_Medals', 'Squad_Size', 'Is_Host', 'EWMA_Score']].to_string())

# 检查中国的数据
print("\n中国 (China) 近5届数据:")
china_data = df_train[df_train['NOC'] == 'China'].sort_values('Year').tail(5)
print(china_data[['NOC', 'Year', 'Target', 'Lag1_Medals', 'Squad_Size', 'Is_Host', 'EWMA_Score']].to_string())

# 检查东道主效应
print("\n东道主国家奖牌数对比:")
host_effect = df_train[df_train['Is_Host'] == 1][['NOC', 'Year', 'Target', 'Lag1_Medals']].copy()
host_effect['Host_Boost'] = host_effect['Target'] - host_effect['Lag1_Medals']
print(host_effect.sort_values('Year').tail(10).to_string())
