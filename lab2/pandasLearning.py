import pandas as pd

# 创建一个字典数据
data = {
    '产品': ['手机', '笔记本', '平板', '耳机', '智能手表', '相机', '电视', '音响', '打印机', '路由器'],
    '销售量': [150, 80, 90, 200, 120, 60, 50, 130, 70, 40],
    '价格(元)': [3000, 6000, 2000, 800, 1500, 5000, 4000, 1500, 1000, 600],
    '销售日期': pd.date_range(start='2023-01-01', periods=10, freq='D')
}

# 创建DataFrame
df = pd.DataFrame(data)

# 输出DataFrame
print(df)

# 筛选出销售量大于100的产品
print("\n")
print("以下是销售量大于100的产品:")
filtered_df = df[df['销售量'] > 100]
print(filtered_df)

# 按销售量从大到小排序
print("\n")
print("以下是按销售量从大到小排序的产品:")
sorted_df = df.sort_values(by='销售量', ascending=False)
print(sorted_df)

# 计算平均销售量
average_sales = df['销售量'].mean()
print(f"平均销售量: {average_sales}")

# 计算每种产品的总销售额，并添加为新列
print("\n")
print("接下来计算每种铲平的总销售额，并添加为新列:")
df['总销售额'] = df['销售量'] * df['价格(元)']
print(df)
