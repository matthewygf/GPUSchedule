WIPfrom collections import OrderedDict
from core import util

class Rack(object):
    """A group of nodes"""
    def __init__(self, rack_id, bandwidth):
        self.rack_id = rack_id
        self.nodes = OrderedDict()
        self.network_bandwidth = bandwidth

    def add_node(self, node):
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
        else:
            util.print_fn("Node already in rack", util.LOG_LEVEL_WARNING)
