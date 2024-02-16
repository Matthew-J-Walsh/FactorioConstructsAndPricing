import json
import logging
logging.basicConfig(level=logging.INFO)
from utils import *
from generators import *
from linearconstructs import *
from linearsolvers import *
from datarawparse import *



class FactorioInstance():
    """
    Holds the information in an instance (specific game mod setup) after completing premanagment steps.

    
    """
    data_raw = None
    constructs = None
    families = None
    reference_list = None
    def __init__(self, filename):
        with open(filename) as f:
            self.data_raw = json.load(f)

        complete_premanagement(self.data_raw)
        self.constructs = generate_all_constructs(self.data_raw)
        self.families, self.reference_list = generate_all_construct_families(self.constructs)  


class FactorioFactoryChain():
    """
    A chain of optimal factory designs starting from the minimal science to a specified point. There are two avaiable types of factory in the chain:
        A science factory that produces a set of science packs. Having one of these will permit any recipe that is unlockable with those sciences to be used in factories after it.
        A component factory that produces a set of buildings. Having one of these will set the pricing model for every factory in the chain up until the next component factory.
    """


