import os 
import csv
from infra import node as n
from infra import rack as r
from core import util

keys_default = [
    'num_switch',
    'num_node_p_switch',
    'num_gpu_p_node',
    'num_cpu_p_node',
    'mem_p_node'
]

class Infrastructure(object):
    """
    NOTE: assumption:
    1. num cpu per node is the same
    2. num gpu per node is the same
    3. mem per node is the same
    4. bandwidth of the rack is the same
    5. num machine per rack is the same
    """
    def __init__(self, flags):
        self.flags = flags
        self.racks = list()
        self.nodes = list()
        self.num_switch = self.flags.num_switch
        self.rack_bandwidth = self.flags.rack_bandwidth
        self.num_nodes_p_switch = self.flags.num_node_p_switch
        self.num_cpu_p_node = self.flags.num_cpu_p_node
        self.num_gpu_p_node = self.flags.num_gpu_p_node
        self.mem_p_node = self.flags.mem_p_node
        self._setup_nodes(flags.cluster_spec)

    def _setup_nodes(self, file_path):
        """read from a csv to init infrastructure"""
        if not os.path.exists(file_path):
            assert ValueError()
        
        project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        spec_file = os.path.join(project_dir, file_path)

        name, ext = os.path.splitext(spec_file)
        # assume it is csv anyway
        assert 'csv' in ext
        f_handler = open(spec_file, 'r')
        reader = csv.DictReader(f_handler, delimiter=',')
        keys = reader.fieldnames
        util.print_fn(keys)

        for default_k in keys_default:
            if default_k not in keys: return
        
        # 1 line after reading fields
        assert reader.line_num == 1

        for row in reader:
            self.num_switch = int(row['num_switch'])
            self.num_nodes_p_switch = int(row['num_node_p_switch'])
            self.num_gpu_p_node = int(row['num_gpu_p_node'])
            self.num_cpu_p_node = int(row['num_cpu_p_node'])
            self.mem_p_node = int(row['mem_p_node'])
        f_handler.close()

        for rack_id in range(0, self.num_switch):
            rack = r.Rack(rack_id, self.rack_bandwidth)
            for node_id in range(0, self.num_nodes_p_switch):
                node = n.Node(node_id, self.num_cpu_p_node, self.num_gpu_p_node, self.mem_p_node)
                self.nodes.append(node)
                rack.add_node(node)
            self.racks.append(rack)

        util.print_fn("num_racks in cluster: %d" % len(self.racks))
        util.print_fn("num_node_p_rack in cluster: %d" % len(self.racks[-1].nodes))
        util.print_fn("num_gpu_p_node in cluster: %d" % self.racks[-1].nodes[-1].gpu_count)
        util.print_fn("num_cpu_p_node in cluster: %d" % self.racks[-1].nodes[-1].cpu_count)
        util.print_fn("mem_p_node in cluster: %d" %  self.racks[-1].nodes[-1].mem_size)
        util.print_fn('--------------------------------- End of cluster spec ---------------------------------')

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

    def get_free_nodes(self):
        return [node for node in self.nodes if node.is_free()]

    def get_available_mem_size(self):
        result = 0
        for node in self.nodes:
            result += node.mem_free()

        return result
