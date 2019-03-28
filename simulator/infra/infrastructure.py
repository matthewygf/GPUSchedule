
class Infrastructure(object):
    def __init__(self, flags):
        self.flags = flags
        self.racks = list()
        self.nodes = list()
        self._setup_nodes(flags)

    def _setup_nodes(self, flags):
        for rack_id in range(0, flags.num_switch):
            rack = Rack(rack_id, flags.rack_bandwidth)
            for node_id in range(0, flags.num_nodes_p_switch):
                node = Node(node_id, flags.num_cpu_p_node, flags.num_gpu_p_node, flags.mem_p_node)
                self.nodes.append(node)
                rack.add_node(node)

            self.racks.append(rack)

    def get_available_cpu_count(self):
        result = 0
        for node in self.nodes:
            result += node.cpu_free()

        return result

    def get_total_gpus(self):
        result = 0
        for node in self.nodes:
            result += node.gpu_count
        return result
        
    def get_free_gpus(self):
        result = 0
        for node in self.nodes:
            result += node.gpu_free()

        return result

    def get_available_mem_size(self):
        result = 0
        for node in self.nodes:
            result += node.mem_free()

        return result
