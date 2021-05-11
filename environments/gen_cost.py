from utils.process import Data

data = Data()
if(data.readData()):
    data.preData()
data.calCost()