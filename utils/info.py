from typing import Dict
from numpy import float64
import yaml

from utils.utils import *
from utils.cost import Cost

Point = Tuple[float, float]

class EnvInfo(object):
    '''
    Generate some info about env information.

    Attributes:
        no_to_name: A Dict from department no to department name.
        no_to_location: A Dict from department no to department location.
        velocity: A float representing postman's delivering velocity.
        ratio: A float used to transform weight to hour. 
    '''

    no_to_name: dict
    no_to_location: Dict[str, Point] 
    velocity: float
    ratio: float

    def __init__(self) -> None:
        super().__init__()
        # read location of original department's information from config.yaml.
        with open('config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)

        records = read_cx(configs['department'])
        for record in records[['机构代码','机构简称','lat','lng']].values:
            self.no_to_name[record[0]] = record[1]
            self.no_to_location[record[0]] = (record[2],record[3])
        
        self.velocity = configs['velocity']
        self.ratio = configs['ratio']

env_info = EnvInfo()
cost = Cost()