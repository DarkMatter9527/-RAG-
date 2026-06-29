import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，确保标签正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义参数和变量范围
s = 2  # 可修改为任意s>1的值，例如3、4等
d=1024
x = np.linspace(0, d/2, 1000)  # 生成(0,32)区间的1000个点，更平滑
y = s **(-2 * x /d)  # 计算函数值

# 创建画布
plt.figure(figsize=(8, 6))

# 绘制曲线
plt.plot(x, y, color='blue', linewidth=2)

# 添加标题和标签
#plt.title(f'函数$y = s^{-2x/64}$的曲线（s={s}，x∈(0,32)）', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 设置坐标轴范围（可选，根据需要调整）
plt.xlim(0, d/2)
plt.ylim(0, 1.1)  # y的最大值为1（x=0时），稍放大范围更美观

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()