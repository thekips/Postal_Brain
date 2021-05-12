import os
import re
import yaml

from .utils import *
from .cost import Cost

class Data:
    '''
    数据读取与处理
    '''
    def __init__(self):
        # 读取参数
        with open('config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        self.velocity = configs['velocity']
        self.ratio = configs['ratio']
        self.department = configs['department']
        self.select = configs['select']
        self.partition = configs['partition']
        self.id2name = configs['id2name']
        self.weilan = configs['weilan']
        self.weilans = configs['weilans']
        
        self.utils = Utils(self.department, self.select, self.partition, self.weilan, self.weilans)
      
    def readData(self):
        '''
        如果不存在处理后的数据，则读取数据并进行预处理
        False -- 数据不需要进行预处理
        True -- 数据需要进行预处理
        '''
        print("确保揽收数据输入格式：投递日期,邮件号,业务种类,重量,投递机构,投递员,投递地址,客户号,lng,lat")
        print("确保投递数据输入格式：投递日期,邮件号,业务种类,重量,投递机构,投递员,道段,投递地址,投递方式,lng,lat")
        print("正在读取揽收数据，投递数据...")
        if os.path.exists('data/lanshou_.csv') and os.path.exists('data/toudi_.csv'):
            return False
        else:
            self.lanshou = readCsv('data/lanshou.csv',False)
            self.toudi = readCsv('data/toudi.csv')
            print("读取完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
            return True

    def preData(self):
        '''
        对数据进行预处理，之后保存至'lanshou_.csv'和'toudi_.csv'两个文件中
        '''
        # 机构文件之外记录
        while True:
            inputs = input("是否筛除机构文件之外的记录？\n"+
                            "是否筛除机构管理员的记录？\n"+
                            "是否筛除周末的记录？\n"+
                            "是否筛除围栏之外的记录？\n"+
                            "是否在街道中进行随机分布？\n"+
                            "是否对不完整地址进行随机分布？\n"+
                            "请输入(y/n)，以空格符号分开：")
            inputs = re.split('\s+', inputs)
            if len(inputs) == 6: break
        
        # 输入要剔除的年份
        if inputs[2] == 'y':
            years = input("请输入要剔除的年份，多个年份之间以空格分开：\n")
            years = re.split('\s+', years)

        # 保存自提点需要的数据
        if not os.path.exists('data/data_step1.csv'):
            print("正在保存自提点需要的中间数据...")
            toudi = self.toudi.copy(deep=True)
            print("中间数据去除大宗")
            toudi = self.utils.subDepartment(toudi)
            print("中间数据去除机构管理员")
            toudi = self.utils.subAdmin(toudi)
            print("中间数据去除围栏")
            toudi = self.utils.subWeilan(toudi)
            self.dicts, self.sets = self.utils.genStreet(toudi)
            print("中间数据随机分布街道")
            temp = toudi[['投递地址','lng','lat']].apply(self.ranStreet,axis=1)
            toudi['lng'] = temp['lng'].values
            toudi['lat'] = temp['lat'].values
            print("中间数据随机分布不完整地址")
            temp = toudi.groupby(by=['投递机构','投递员','投递日期'],as_index=False)
            temp = temp[['投递地址','lng','lat']]
            temp = temp.apply(self.ranAbnormal)
            temp.index = temp.index.droplevel()
            toudi.loc[temp.index, 'lng'] = temp['lng'].values
            toudi.loc[temp.index, 'lat'] = temp['lat'].values
            print("中间数据去除围栏")
            toudi = self.utils.subWeilan(toudi)
            toudi = toudi.drop(toudi[toudi['lng'].isnull()].index)
            # 保存自提点需要的中间数据
            toudi.to_csv('data/data_step1.csv', encoding='gb18030', index=False)
            del toudi
            print("已保存自提点需要的中间数据")

        # 大宗机构
        if inputs[0] == 'y':
            print("正在筛除机构文件之外的记录...")
            self.lanshou = self.utils.subDepartment(self.lanshou)
            self.toudi = self.utils.subDepartment(self.toudi)
            print("筛除完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        else:
            print("不筛除机构文件之外的记录")

        # 机构管理员记录
        if inputs[1] == 'y':
            print("正在筛除机构管理员的记录...")
            self.lanshou = self.utils.subAdmin(self.lanshou)
            self.toudi = self.utils.subAdmin(self.toudi)
            print("筛除完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        else:
            print("不筛除机构管理员的记录")

        # 周末记录
        if inputs[2] == 'y':
            print("正在筛除周末的记录...\n", years)
            self.lanshou = self.utils.subHoliday(self.lanshou, years)
            self.toudi = self.utils.subHoliday(self.toudi, years)
            print("筛除完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        else:
            print("不筛除周末的记录")

        # 围栏之外记录
        if inputs[3] == 'y':
            print("正在筛除围栏之外的记录...")
            self.lanshou = self.utils.subWeilan(self.lanshou)
            self.toudi = self.utils.subWeilan(self.toudi)
            print("筛除完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        else:
            print("不筛除围栏之外的记录")

        # 街道中进行随机分布，按街道范围对不完整街道地址分布
        if inputs[4] == 'y':
            print("正在街道中进行随机分布...")
            # 处理lanshou数据
            self.dicts, self.sets = self.utils.genStreet(self.lanshou)
            temp = self.lanshou[['投递地址','lng','lat']].apply(self.ranStreet,axis=1)
            self.lanshou['lng'] = temp['lng'].values
            self.lanshou['lat'] = temp['lat'].values
            # 处理toudi数据
            self.dicts, self.sets = self.utils.genStreet(self.toudi)
            temp = self.toudi[['投递地址','lng','lat']].apply(self.ranStreet,axis=1)
            self.toudi['lng'] = temp['lng'].values
            self.toudi['lat'] = temp['lat'].values
            print("随机分布完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        else:
            print("不在街道中进行随机分布")

        # 不完整地址进行随机分布，按揽投员某时段（上/下午）对不完整地址进行分布
        if inputs[5] == 'y':
            print("正在对不完整地址进行随机分布...")
            # 处理lanshou数据
            temp = self.lanshou.groupby(by=['投递机构','投递员','投递日期'],as_index=False)
            temp = temp[['投递地址','lng','lat']]
            temp = temp.apply(self.ranAbnormal)
            temp.index = temp.index.droplevel()
            self.lanshou.loc[temp.index, 'lng'] = temp['lng'].values
            self.lanshou.loc[temp.index, 'lat'] = temp['lat'].values
            # 处理toudi数据
            temp = self.toudi.groupby(by=['投递机构','投递员','投递日期'],as_index=False)
            temp = temp[['投递地址','lng','lat']]
            temp = temp.apply(self.ranAbnormal)
            temp.index = temp.index.droplevel()
            self.toudi.loc[temp.index, 'lng'] = temp['lng'].values
            self.toudi.loc[temp.index, 'lat'] = temp['lat'].values
            print("随机分布完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))
        else:
            print("不对不完整地址进行随机分布")

        # 筛除围栏外记录和空白经纬度记录
        print("正在筛除围栏之外的记录...")
        self.lanshou = self.utils.subWeilan(self.lanshou)
        self.toudi = self.utils.subWeilan(self.toudi)
        self.lanshou = self.lanshou.drop(self.lanshou[self.lanshou['lng'].isnull()].index)
        self.toudi = self.toudi.drop(self.toudi[self.toudi['lng'].isnull()].index)
        print("筛除完成，揽收数据共%d条记录，投递数据共%d条记录" %(len(self.lanshou),len(self.toudi)))

        # 设定揽投，重量，地址类型难度系数
        print("正在进行系数设定...")

        # 设定邮件类型分类
        self.lanshou = self.utils.makeType(self.lanshou)
        self.toudi = self.utils.makeType(self.toudi)

        # 设定揽投系数
        # 揽收 -- 按客户代码设定系数
        self.lanshou.loc[self.lanshou['客户号'].isnull(), '业务种类'] = 5
        self.lanshou.loc[self.lanshou['客户号'] != 5, '业务种类'] = 1.5
        # 投递 -- 按投递方式与业务种类设定系数
        self.toudi.loc[self.toudi['业务种类'].isnull(),'业务种类'] = '快包' # 为业务种类中未标注的标注为快包
        # 处理退回妥投
        abnormal = self.toudi[self.toudi['投递方式'].str.contains('退回妥投', na=False)].copy()
        abnormal.loc[abnormal['业务种类']=='快包', '业务种类'] = 1.2
        abnormal.loc[abnormal['业务种类']!=1.2, '业务种类'] = 1.4
        # 处理非退回妥投
        normal = self.toudi.drop(abnormal.index).copy()
        normal.loc[normal['业务种类'].str.contains(r'快包|其他', na=False), '业务种类'] = 1.2
        normal.loc[normal['业务种类'].str.contains(r'国际|标快', na=False), '业务种类'] = 1.5
        normal.loc[normal['业务种类'].str.contains(r'法院', na=False), '业务种类'] = 2
        normal.loc[normal['业务种类'].str.contains(r'高端政务', na=False), '业务种类'] = 4
        normal.loc[normal['邮件号'].str.contains(r'^1.*8[0-9]$', na=False), '业务种类'] = 2
        normal.loc[normal['邮件号'].str.contains(r'^1.*9[2-4]$', na=False), '业务种类'] = 2
        # 退回、非退回均写入投递
        self.toudi.loc[abnormal.index, '业务种类'] = abnormal['业务种类']
        self.toudi.loc[normal.index, '业务种类'] = normal['业务种类']

        # 设定重量系数
        self.lanshou = self.utils.makeWeight(self.lanshou)
        self.toudi = self.utils.makeWeight(self.toudi)

        # 设定地址类型难度系数
        self.lanshou = self.utils.makeHard(self.lanshou)
        self.toudi = self.utils.makeHard(self.toudi)

        # 设定揽收地址中机构地址的经纬度
        self.lanshou = self.utils.makeSpecial(self.lanshou)

        # 计算曼哈顿距离
        self.lanshou = self.utils.calDistance(self.lanshou)
        self.toudi = self.utils.calDistance(self.toudi)

        # 用id值标识数据是揽收还是投递的
        self.lanshou['id'] = True
        self.toudi['id'] = False

        # 揽投员字段只留下工号
        self.lanshou['投递员'] = self.lanshou['投递员'].str.split(' ',expand=True)[0].values
        self.lanshou['投递员'].apply(str)
        self.toudi['投递员'] = self.toudi['投递员'].str.split(' ',expand=True)[1].values
        self.toudi['投递员'].apply(str)

        # 去除具体时间改为上下午，增加每件的cost列
        self.lanshou['投递日期'] = self.lanshou['投递日期'].apply(lambda x: x[:-8] +('下午' if x[-8:] > '13:30' else '上午'))
        self.toudi['投递日期'] = self.toudi['投递日期'].apply(lambda x: x[:-8] +('下午' if x[-8:] > '13:30' else '上午'))
        self.lanshou['cost'] = self.lanshou['业务种类'] * self.lanshou['重量'] * self.lanshou['type']
        self.toudi['cost'] = self.toudi['业务种类'] * self.toudi['重量'] * self.toudi['type']

        # 设定揽投，重量，地址类型难度系数完毕
        print("系数设定完毕")

        # 保存处理后的数据
        print("正在保存处理后的数据到新文件中...")
        self.lanshou.to_csv('data/lanshou_.csv', encoding='gb18030', index=False)
        self.toudi.to_csv('data/toudi_.csv', encoding='gb18030', index=False)
        del self.lanshou
        del self.toudi
        print("保存完毕")

    def calCost(self):
        '''
        调用cost类进行代价计算
        calRes将结果保存至'res_man.csv'和'res_all.csv'两个文件中
        '''
        self.genDict() # 生成机构代码到机构简称、经纬度的映射，揽投员id到揽投员姓名的映射
        self.cost = Cost(self.velocity, self.ratio, self.noToName, self.noToLat, self.noToLng, self.id2Name)
        self.cost.calRes() # 更改path以更改文件前缀

    def genDict(self):
        '''
        生成机构代码到机构简称、经纬度的映射
        揽投员id到揽投员姓名的映射
        '''
        self.noToName = {} # 机构代码到机构简称
        self.noToLat = {} # 机构代码到机构纬度
        self.noToLng = {} # 机构代码到机构经度
        self.id2Name = {} # 揽投员id到揽投员姓名

        records = readCX(self.department)
        for record in records[['机构代码','机构简称']].values:
            self.noToName[record[0]] = record[1]

        records = readCX(self.select)
        for record in records[['机构代码','lat','lng']].values:
            self.noToLat[record[0]] = record[1]
            self.noToLng[record[0]] = record[2]

        records = readCX(self.id2name)
        for record in records[['揽投员工号','作业人员姓名']].dropna().values:
            self.id2Name[record[0]] = record[1]

    def ranStreet(self, x):
        '''
        针对出现在set中的只精确到街道的地址，由dicts为其取经纬度随机值
        '''
        if x['投递地址'] in self.sets:
            addr = x['投递地址']
            lng = self.dicts.loc[addr]['lng']
            lat = self.dicts.loc[addr]['lat']
            x['lng'] = np.random.random() * (lng['amax'] - lng['amin']) + lng['amin']
            x['lat'] = np.random.random() * (lat['amax'] - lat['amin']) + lat['amin']
        
        return x

    def ranAbnormal(self, x):
        '''
        针对以省市区县结尾的地址，由揽投员某时段（上/下午）所有范围对不完整地址进行分布
        '''
        index = x[x['投递地址'].str.contains(r'[省市区县]$', na=False)].index#地址数据不完整的记录
        index = index.append(x[x['lng'].isnull()].index).sort_values()#经纬度为空的记录
        todo = x.loc[index]
        done = x.drop(todo.index)

        if len(todo) < 1 or len(done) < 1: return

        maxy = done['lng'].max()
        miny = done['lng'].min()
        maxx = done['lat'].max()
        minx = done['lat'].min()
        lng = np.random.random(len(todo)) * (maxy - miny) + miny
        lat = np.random.random(len(todo)) * (maxx - minx) + minx

        x.loc[todo.index, 'lng'] = lng
        x.loc[todo.index, 'lat'] = lat
        
        return x

if __name__ == '__main__':

    data = Data()
    if data.readData(): # 若数据需要预处理
        data.preData()
    data.calCost()