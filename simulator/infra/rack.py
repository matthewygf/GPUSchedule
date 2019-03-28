from core import util

class Rack(object):
    """A group of nodes"""
    def __init__(self, rack_id, bandwidth):
        self.rack_id = rack_id
        self.nodes = list()
        self.network_bandwidth = bandwidth

    def __eq__(self, other):
        result = (self.rack_id == other.rack_id)
        return result

    def add_node(self, node):
        if not self.nodes.__contains__(node):
            self.nodes.append(node)
        else:
            util.print_fn("Node already in rack", util.LOG_LEVEL_WARNING)
