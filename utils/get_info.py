from typing import Dict
import yaml

from utils.utils import *

class DepInfo(object):
    '''
    Generate some info about original derpartment

    Attributes:
        no_to_name: A Dict from department no to department name.
        no_to_lat: A Dict from department no to department latitude.
        no_to_lng: A Dict from department no to department longitude.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.no_to_name = {}
        self.no_to_lat = {}
        self.no_to_lng = {}
        # read location of file of original department's information from config.yaml.
        with open('config.yaml', 'r', encoding='utf-8') as f:
            department = yaml.safe_load(f)['department']

        records = read_cx(department)
        for record in records[['机构代码','机构简称','lat','lng']].values:
            self.no_to_name[record[0]] = record[1]
            self.no_to_lat[record[0]] = record[2]
            self.no_to_lng[record[0]] = record[3]

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