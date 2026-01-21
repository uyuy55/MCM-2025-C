# MCM-2025-C
### **2. 数据预处理 (Data Processing)**

* 
这部分对于本题至关重要，因为数据量大且杂 。


* 
**2.1 数据概览：** 简述使用了提供的5个数据集 。


* **2.2 数据清洗 (Data Cleaning):**
* **关键点：** 解释如何处理国家代码变更（如苏联->俄罗斯，西德/东德）。说明如何合并 `athlete` 数据来计算各国的项目优势。
* *分工提示：* 队员 B 负责写这段，描述处理细节。



### **3. 模型一：奖牌榜宏观预测模型 (Model I: Macro-Prediction for Medal Tables)**

* *目标：解决 Q1 (2028预测), Q2 (零突破)*
* **3.1 特征工程 (Feature Engineering):**
* 定义影响因子：历史成绩（动量）、是否东道主（Host）、距离上次举办的时间、项目总数（Events Count）。


* **3.2 模型建立 (Model Construction):**
* 描述使用的预测模型（例如：Time Series Clustering, Gradient Boosting, or Poisson Regression）。
* 
**重点：** 必须包含**不确定性分析**（Prediction Intervals）。




* **3.3 结果分析 (Results):**
* 
**2028 洛杉矶预测：** 给出 Top 10 榜单。明确指出美国（Home）的涨幅 。


* 
**进步与退步者：** 回答 "Which countries most likely to improve/worse" 。




* **3.4 零的突破 (Breaking the Zero):**
* 建立一个二分类模型（Logistic Regression），预测未获奖国家  的概率。
* 给出具体的国家名单和概率估算 。




* *分工提示：* 队员 A 主笔。

### **4. 模型二：微观影响因子分析 (Model II: Micro-Analysis of Factors)**

* *目标：解决 Q3 (项目), Q4 (教练)*
* **4.1 项目设置与东道主效应 (Events & Host Effect):**
* 分析 `summerOly_programs.csv`。
* 回答：东道主是否通过增加自己擅长的项目（Event Types）来获利？


* 可视化：展示项目数量变化与东道主奖牌增量的相关性。


* **4.2 伟大的教练效应 (The "Great Coach" Effect):**
* **难点高光区：** 定义“教练效应”的数学表达。
* 模型假设：由于数据没有直接教练列，假设“成绩突变”或“跨国且成绩带入”为教练效应。
* 
**案例分析：** 利用 Lang Ping 或 Béla Károlyi 的数据点，验证模型 。


* 
**量化结果：** 估计引进一个顶级教练能带来多少奖牌期望值的提升 。




* *分工提示：* 队员 B 主笔。

### **5. 策略应用与建议 (Strategic Applications)**

* *目标：解决 Q4 (投资建议), Q5 (Insight)*
* **5.1 投资建议 (Investment Plan):**
* 基于模型二，挑选 3 个国家+项目组合（例如：建议印度投资射箭教练），并预测潜在回报 。




* **5.2 给奥组委的建议 (Recommendations):**
* 基于模型发现的规律（Insight），提出 2-3 条具体建议 。





### **6. 敏感性分析与模型评价 (Sensitivity Analysis & Evaluation)**

* **6.1 鲁棒性检验：** 如果改变历史数据的权重（比如只看最近3届 vs 最近5届），预测结果是否稳定？
* **6.2 优缺点分析 (Strengths & Weaknesses):**
* 优点：考虑了微观因素、有置信区间。
* 缺点：无法完全量化教练的非技术影响（如心理辅导）、未考虑地缘政治影响。
