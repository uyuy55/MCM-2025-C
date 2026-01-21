import pandas as pd
df = pd.read_csv('df_train_panel.csv')

print('=' * 60)
print('奖牌效率 (Efficiency) = Medals / Squad_Size')
print('=' * 60)

print('\n【统计摘要】')
print(df['Efficiency'].describe().round(3))

print('\n【2024年Top15国家效率排名】')
df_2024 = df[df['Year'] == 2024].dropna(subset=['Efficiency'])
df_2024 = df_2024.sort_values('Efficiency', ascending=False)
print(df_2024[['NOC', 'Target', 'Squad_Size', 'Efficiency']].head(15).to_string(index=False))

print('\n【历史效率最高的记录 (Squad_Size>=50)】')
df_high = df[df['Squad_Size'] >= 50].nlargest(10, 'Efficiency')
print(df_high[['NOC', 'Year', 'Target', 'Squad_Size', 'Efficiency']].to_string(index=False))

print('\n【主要国家2024效率对比】')
major = ['United States', 'China', 'Great Britain', 'France', 'Japan', 'Australia', 'Germany']
df_major = df[(df['Year'] == 2024) & (df['NOC'].isin(major))]
df_major = df_major.sort_values('Efficiency', ascending=False)
print(df_major[['NOC', 'Target', 'Squad_Size', 'Efficiency']].to_string(index=False))
