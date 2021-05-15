## 邮政大脑

邮政大脑项目的后续工作，采用强化学习来解决揽投机构的选址问题。


### Environment

pip install numpy pandas Pyyaml arrow shapely

| 环境          | 模块           |
| ------------- | -------------- |
| python==2.8.1 | arrow==0.17.0  |
|               | numpy==1.19.5  |
|               | pandas==1.2.1  |
|               | PyYAML==5.4.1  |
|               | Shapely==1.7.1 |

### Input & Output

处理之前数据输入要求如下（投递数据会多出一个道段字段）：

|     投递日期 | 邮件号 | 业务种类 |     重量 | 投递机构 |                           投递员 | 投递地址 | 客户号 |  lng |  lat |
| -----------: | -----: | -------: | -------: | -------: | -------------------------------: | -------: | -----: | ---: | ---: |
| 年月日时分秒 |        |   标快等 | kg为单位 | 机构代码 | 揽收为姓名+代码，投递为代码+姓名 |          |        | 经度 | 纬度 |

处理之后数据输出字段如下（投递数据会多出一个道段字段）：

|   投递日期   | 邮件号 | 业务种类 | 重量 | 投递机构 |   投递员   | 投递地址 | 客户号 | lng  | lat  |  分类  | dist | type |  cost  |           id            |
| :----------: | :----: | :------: | ---- | :------: | :--------: | :------: | :----: | :--: | :--: | :----: | :--: | :--: | :----: | :---------------------: |
| 年月日上下午 |        |   权重   | 权重 | 机构代码 | 揽投员代码 |          |        | 经度 | 纬度 | 计数用 |      | 权重 | 权重积 | 揽收为True，投递为False |

### Data processing augment

数据处理的参数设置在''config.yaml''文件中修改，可修改参数如下：

```python
('velocity', type=int, default=10, help='Velocity: 揽投员运行速度')
('ratio', type=int, default=24, help='Ratio: 将系数转换为时间/h,设置为24则有平均一件0.05h')
('department', type=str, default='data/江门机构.xlsx', help='Department: 需要处理的机构部门文件路径')
('select', type=str, default='data/江门机构.xlsx', help='Select: 机构部门经纬度文件的路径')
('partition', type=str, default='data/江门区域划分表.xls', help='Select: CBD等区域划分文件的路径')
('id2name', type=str, default='data/工号-作业姓名.xlsx', help='Id2name: 工号-作业姓名文件路径')
('weilan', type=str, default='data/机构围栏表.xlsx', help="Weilan: 机构围栏表文件路径")
('weilans', type=str, default='data/机构围栏坐标点表.csv', help="Weilan: 机构围栏坐标点表文件路径")
```

补充：基础快包一个件算出来权重1.2，1.2 / 24=0.5h，也就是平均一件3min。