import neuro
import queue

class Event:
    def __init__(self, t, n, v, tcheck=False):
        self.t = int(t)
        self.node = n
        self.val = v
        self.tcheck = tcheck

    def values(self):
        return (self.t, self.node, self.val)

    def __lt__(self, other):
        return self.t < other.t

class PyProcessor(neuro.Processor):

    def __init__(self, config):
        neuro.Processor.__init__(self)
        self.config = config
        self.network = None
        self.properties = neuro.PropertyPack()

        self.queue = queue.PriorityQueue()
        self.charges = dict()
        self.tcheck = dict()
        self.spike_cnt = list()
        self.spike_t = list()

        self.t_id = self.properties.add_node_property("Threshold", 0, 1)
        self.w_id = self.properties.add_edge_property("Weight", -1, 1)
        self.d_id = self.properties.add_edge_property("Delay", 0, 8, neuro.PropertyType.Integer)

    def load_network(self, network, net_id = 0):
        self.network = network
        self.clear_activity()

    def load_networks(self, networks):
        raise NotImplementedError("Loading multiple networks is not supported for PyProcessor")

    def apply_spike(self, spike, net_id = 0):
        node = self.network.get_input(spike.id)
        self.queue.put(Event(spike.time, node, spike.value))

    def apply_spikes(self, spikes, net_id = 0):
        for spike in spikes:
            node = self.network.get_input(spike.id)
            self.queue.put(Event(spike.time, node, spike.value))

    def run(self, duration, net_id = 0):
        # Calculate when to stop
        end_time = self.cur_time + duration

        while not self.queue.empty() and self.cur_time < end_time:
            e = self.queue.get()

            # if the event is later than the end time, add it back and quit
            if e.t > end_time:
                self.queue.put(e)
                break

            t, node, val = e.values()
            idx = node.id

            # Update time
            self.cur_time = t

            # Process threshold check events
            if e.tcheck:

                self.tcheck[idx] = False

                # Check if it is firing
                if self.charges[idx] > node.get(self.t_id):
                    # Reset charge
                    self.charges[idx] = 0

                    # Check if output
                    if node.output_id >= 0:
                        self.spike_cnt[node.output_id] += 1
                        self.spike_t[node.output_id] = self.cur_time

                    # Add events
                    for edge in node.outgoing:
                        self.queue.put(Event(self.cur_time + edge.get(self.d_id) + 1, 
                                             edge.post, 
                                             edge.get(self.w_id)))
            else:
                # Accumulate charge
                if idx not in self.charges:
                    self.charges[idx] = val
                    self.tcheck[idx] = False
                else:
                    self.charges[idx] += val

                # Make threshold check event
                if not self.tcheck[idx] and self.charges[idx] > node.get(self.t_id):
                    self.queue.put(Event(self.cur_time + 1, node, 0, True))
                    self.tcheck[idx] = True

        self.cur_time = end_time

    def get_time(self, net_id = 0):
        return self.cur_time

    def track_aftertime(self, output_id, aftertime, net_id = 0):
        pass

    def track_output(self, track, net_id=0):
        pass

    def output_last_fire(self, output_id, net_id = 0):
        return self.spike_t[output_id]

    def output_count(self, output_id, net_id = 0):
        return self.spike_cnt[output_id]

    def output_vector(self, output_id, net_id = 0):
        raise NotImplementedError("Output spike vectors are not supported in PyProcessor")

    def clear(self, net_id = 0):
        self.network = None
        self.clear_activity()

    def clear_activity(self, net_id = 0):
        self.spike_cnt = [0 for _ in range(self.network.num_outputs())] if self.network is not None else list()
        self.spike_t = [-1 for _ in range(self.network.num_outputs())] if self.network is not None else list()
        self.tcheck = dict()
        self.charges = dict()
        self.queue = queue.PriorityQueue()
        self.cur_time = 0

    def get_properties(self):
        return self.properties
