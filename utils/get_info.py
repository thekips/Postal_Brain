from typing import Dict
import yaml

from utils.utils import *

class DepInfo(object):
    '''
    Generate some info about original derpartment

    Attributes:
        no_to_name: A Dict from department no to department name.
        no_to_location: A Dict from department no to department location.
    '''

    def __init__(self) -> None:
        super().__init__()
        # read location of original department's information from config.yaml.
        with open('config.yaml', 'r', encoding='utf-8') as f:
            department = yaml.safe_load(f)['department']
        self.no_to_name = {}
        self.no_to_location = {}
        records = read_cx(department)
        for record in records[['机构代码','机构简称','lat','lng']].values:
            self.no_to_name[record[0]] = record[1]
            self.no_to_location[record[0]] = (record[2],record[3])

class EnvInfo(object):
    '''
    Generate some info like postman's velocity, ratio which is used to transform weight to hour.

    Attributes:
       velocity: A float representing postman's delivering velocity.
       ratio: A float used to transform weight to hour. 
    '''

    def __init__(self) -> None:
        super().__init__()

        with open('config.yaml', 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        self.velocity = configs['velocity']
        self.ratio = configs['ratio']