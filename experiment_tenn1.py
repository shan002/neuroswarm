# from multiprocessing import Pool, TimeoutError
from tqdm.contrib.concurrent import process_map
import neuro
import caspian
import random
import argparse
import os
import time
import pathlib
# import matplotlib.pyplot as plt


# Provided Python utilities from tennlab framework/examples/common
from common.evolver import Evolver
from common.evolver import MPEvolver
import common.utils as nutils
from common.utils import json
from common.application import Application

directory = os.path.dirname(os.path.realpath(__file__))

try:
    import pandas
except ImportError:
    pandas = None


class CustomPool():
    """pool class for Evolver, so we can use tqdm for those sweet progress bars"""

    def __init__(self, **tqdm_kwargs):  # max_workers=args.threads
        self.kwargs = tqdm_kwargs

    def map(self, fn, *iterables):
        return process_map(fn, *iterables, **self.kwargs)


class TennBots(Application):
    """Tennbots application for TennLab neuro framework


    """

    def __init__(self, **kwargs):
        self.n_inputs = 2  # default. set one later in the code if necessary
        self.n_outputs = 2  # default. set one later in the code if necessary

        self.env_name = kwargs['environment']
        self.label = kwargs['label']
        self.seed = kwargs['seed']
        self.viz = kwargs['viz']
        self.agents = kwargs['agents']
        self.prompt = kwargs['prompt']
        self.viz_delay = kwargs['viz_delay']
        self.big_fitness = (kwargs['methodology'] == 'big_run')
        self.one_small = (kwargs['methodology'] == 'one_small')
        self.episodes = kwargs['episodes']
        self.processes = kwargs['processes']
        self.sim_time = kwargs['sim_time']
        self.logfile = kwargs['logfile']

        # And set parameters specific to usage.  With testing,
        # we'll read the app_params from the network.  Otherwise, we get them from
        # the command line or defaults.

        app_params = ['proc_ticks', ]
        self.app_params = dict()

        # Copy network from file into memory, including processor & app params
        if kwargs['action'] == "test":
            with open(kwargs['network']) as f:
                j = json.loads(f.read())
            self.net = neuro.Network()
            self.net.from_json(j)
            self.processor_params = self.net.get_data("processor")
            self.app_params = self.net.get_data("application").to_python()

        # Get params from defaults/cmd params and default proc/eons cfg
        elif kwargs['action'] == "train":
            # self.input_time = kwargs['input_time']
            self.proc_ticks = kwargs['proc_ticks']
            self.training_network = kwargs["training_network"]
            self.eons_params = nutils.load_json_string_file(kwargs["eons_params"])
            self.processor_params = nutils.load_json_string_file(kwargs['processor_params'])
            self.runs = kwargs['runs']

        # If 'test' and an app param hasn't been set, we simply use defaults (we won't be
        # able to read them from the command line).  This is how we can add information to
        # networks, and not have old networks be obsolete.

        for arg in app_params:
            if not (arg in self.app_params):
                self.app_params[arg] = kwargs[arg]
        # self.flip_up_stay = self.app_params['flip_up_stay']

        # Note: encoders/decoders *can* be saved to or read from the network. not implemented yet.

        # for now they are just being used to calculate n_inputs/outputs

        # Setup encoder
        # for each binary raw input, we encode it to constant spikes on bins, kinda like traditional one-hot
        encoder_params = {
            "dmin": [0] * 5,  # two bins for each binary input
            "dmax": [1] * 5,
            "interval": self.app_params['proc_ticks'],
            "named_encoders": {"s": "spikes"},
            "use_encoders": ["s"] * 5
        }
        self.encoder = neuro.EncoderArray(encoder_params)
        self.n_inputs = self.encoder.get_num_neurons()

        # Setup decoder
        # Read spikes to wheelspeeds using rate-based decoding
        decoder_params = {
            # see notes near where decoder is used
            "dmin": [0] * 4,
            "dmax": [self.app_params['proc_ticks']] * 4,
            "divisor": self.app_params['proc_ticks'],
            "named_decoders": {"r": {"rate": {"discrete": True}}},
            "use_decoders": ["r", "r", "r", "r"]
        }
        self.decoder = neuro.DecoderArray(decoder_params)
        self.n_outputs = self.decoder.get_num_neurons()

        # Set up the initial gym, and set up the action space.

        # env = gym.make(self.env_name)

        # self.action_min = [0]
        # self.action_max = [env.action_space.n-1]
        # self.action_type = env.action_space.dtype.type

        self.log("initialized experiment_tenn1")

        # If we're playing from stdin, we can return now -- nothing
        # else is being used

        if kwargs['action'] == "stdin":
            return

    def fitness(self, processor, network):
        return sum(self.run(processor, network) for i in range(self.runs))

    def run(self, processor, network, prerun_callback=lambda sim: None):
        import tennbots
        # setup sim
        rng = random.Random()
        sim = tennbots.Sim(self.agents, rng, render_mode="human" if self.viz else None)
        # setup processors
        pprops = processor.get_configuration()
        # print(pprops)
        processors = [caspian.Processor(pprops)] * self.agents
        for proc in processors:
            proc.load_network(network)
            neuro.track_all_output_events(proc, network)
            # proc.track_neuron_events(0)
        # setup a spot to save actions to
        actions = [(0, 0)] * self.agents
        loss_graph = []

        # save references for loop optimization
        encoder, decoder = self.encoder, self.decoder
        proc_ticks = self.app_params['proc_ticks']

        def b2oh(x: bool):
            return (0, 1) if x else (1, 0)

        def get_action(processor, observation):
            # unpack observation
            ir_sensed, goal_sensed = observation
            # convert each binary to one-hot and concatenate
            input_vector = b2oh(ir_sensed) + b2oh(goal_sensed) + (rng.randint(0, 1), )
            spikes = encoder.get_spikes(input_vector)
            processor.apply_spikes(spikes)
            processor.run(proc_ticks)
            # action: bool = bool(proc.output_vectors())  # old. don't use.
            data = decoder.get_data_from_processor(processor)
            # four bins. Two for each wheel, one for positive, one for negative.
            v = 0.2 * (data[1] - data[0])
            w = 2.0 * (data[3] - data[2])
            return (v, w)

        prerun_callback(sim)

        for i in range(self.sim_time):
            observations, reward, _, _, info, *_ = sim.step(actions)
            loss_graph.append(reward)
            actions = (get_action(p, o) for p, o in zip(processors, observations))
            # print(f"obsv: {observations}\n\n")
            # print(f"act: {actions}\n\n")
            if self.viz:
                sim.render()

        observations, reward, _, _, info, *_ = sim.step(actions)
        # how_many_goal_sensed = sum([goal_sensed for ir_sensed, goal_sensed in observations])

        # reward += (how_many_on_goal ** 2) * 1000

        # loss = sum(loss_graph[-5000:])
        return reward

    def save_network(self, net, safe_overwrite=True):
        path = pathlib.Path(self.training_network)
        if self.label != "":
            net.set_data("label", self.label)
        net.set_data("processor", self.processor_params)
        net.set_data("application", self.app_params)

        if safe_overwrite and path.is_file():
            path.rename(path.with_suffix('.bak'))
        with open(path, 'w') as f:
            f.write(str(net))

        self.log(f"wrote best network to {str(path)}.")

    def log_status(self, info):
        print(info)
        self.log(info)

    def log(self, msg, timestamp="%Y%m%d %H:%M:%S", prompt=' >>> ', end='\n'):
        if not self.logfile:
            return
        if isinstance(timestamp, str) and '%' in timestamp:
            timestamp: str = time.strftime(timestamp)
        with open(self.logfile, 'a') as f:
            f.write(f"{timestamp}{prompt}{msg}{end}")


def train(**kwargs):
    app = TennBots(**kwargs)

    processes = kwargs["processes"]
    epochs = kwargs["epochs"]
    max_fitness = kwargs["max_fitness"]

    if processes == 1:
        evolve = Evolver(
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
        )
    else:
        evolve = MPEvolver(  # multi-process for concurrent simulations
            app=app,
            eons_params=app.eons_params,
            proc_name=caspian.Processor,
            proc_params=app.processor_params,
            pool=CustomPool(max_workers=processes),
        )
    evolve.print_callback = app.log_status

    try:
        evolve.train(epochs, max_fitness)
    except KeyboardInterrupt:
        app.log("training cancelled.")
        raise
    app.log("training finished.")

    # evolve.graph_fitness()


def run(**kwargs):
    app = TennBots(**kwargs)

    # Set up simulator and network

    if kwargs["action"] == "stdin":
        proc = None
        net = None
    else:
        proc = caspian.Processor(app.processor_params)
        net = app.net

    # Run app and print fitness
    print("Fitness: {:8.4f}".format(app.run(proc, net)))


def test(**kwargs):
    app = TennBots(**kwargs)

    # Set up simulator and network
    proc = caspian.Processor(app.processor_params)
    net = app.net

    # Run app and print fitness
    print("Fitness: {:8.4f}".format(app.run(proc, net)))


def main():
    # parse cmd line args and run either `train(...)` or `run(...)`

    parser = argparse.ArgumentParser(description='Freeway app for eons, neuro or stdin')
    parser.add_argument('action', choices=['train', 'test', 'run', 'stdin'])

    # Parameters that apply to all situations.  These are the only ones that I give defaults.

    parser.add_argument('--seed', default=0, type=int, help="[all] rng seed for the app (0)")
    parser.add_argument('--methodology',
                        choices=['big_run', 'small_runs', 'one_small'], default='big_run',
                        help="[all] one big run/10 small runs/1 small run (big_run).")
    parser.add_argument('--episodes', default=1, type=int, help="[all] # of episodes to run (1)")
    parser.add_argument('--agents', default=10, type=int, help="[all] # of agents to run with.")
    parser.add_argument('--show_collisions',
                        help="[all] print whether there is a collision (unset)",
                        action="store_true")
    parser.add_argument('--show_observations', help="[all] print all 128 observations (unset)",
                        action="store_true")
    parser.add_argument('--show_changes', help="[all] print the observations that change (unset)",
                        action="store_true")
    parser.add_argument('--sim_time', type=int, default=9999,
                        help="[train] time steps per simulate() (9999).")

    # Parameters that only apply to test or stdin.
    # Don't use defaults here, because we don't want the user to specify them when they don't apply.

    parser.add_argument('--viz', help="[test,stdin] use game visualizer", action="store_true")
    parser.add_argument('--noviz', help="[test,stdin] use game visualizer", action="store_true")
    parser.add_argument('--viz_delay', type=float,
                        help="[test,stdin] change the delay between timesteps in the viz (0.016)")
    parser.add_argument('--prompt', help="[test] wait for a return to continue at each step.", action="store_true")
    parser.add_argument('--network', help="[test] network file (networks/experiment_tenn1.json)")

    # Parameters that only apply to training - observations and spike encoders.
    # Don't use defaults here, because we don't want the user to specify them when they don't apply.

    parser.add_argument('--label',
                        help="[train] label to put into network JSON (key = label)")
    parser.add_argument('--training_network',
                        help="[train] output network file (networks/experiment_tenn1_train.json)")
    parser.add_argument('--logfile',
                        help="[train] running log file (tenn1_train.log)")

    parser.add_argument('--bias', help="[train] add a bias neuron (unset)", action="store_true")
    parser.add_argument('--flip_up_stay', help="[train] flip output neurons 0 and 1 (unset)",
                        action="store_true")

    parser.add_argument('--decoder',
                        help="[train] json for the SpikeDecoder for player's actions (wta-3)")

    parser.add_argument('--testing_data',
                        help="[test] testing dataset")
    parser.add_argument('--runs', required=False, type=int,
                        help="[train] how many runs are used to calculate fitness for a network")

    # Parameters that only apply to training - all of the other stuff.
    # Again, don't use defaults, because we don't want the user to specify
    # them when they don't apply.

    parser.add_argument('--proc_ticks', type=float,
                        help="[train] time steps per processor output (10).")
    # parser.add_argument('--input_time', type=float,
    #                     help="[train] time steps over which to pulse input (50).")
    parser.add_argument('--eons_params', help="[train] json for eons parameters (eons/std.json)")
    parser.add_argument('--processor_params',
                        help="[train] json for processor parameters (config/caspian.json)")
    parser.add_argument('--processes', type=int, help="[train] # threads (1)")
    parser.add_argument('--max_fitness', required=False, type=int,
                        help="[train] stop eons if this fitness is achieved (34/1)")
    parser.add_argument('--epochs', required=False, type=int,
                        help="[train] training epochs (50)")
    parser.add_argument('--graph', help="[train] graph eons results", action="store_true")
    args = parser.parse_args()

    config = vars(args)

    # Go through the pain of error checking the command line, so that users don't think that they
    # are setting parameters that they are not.  Also set defaults here.

    if args.action == "train":
        illegal = ['viz', 'viz_delay', 'network', 'prompt']
        for s in illegal:
            if config[s]:
                print(f"Cannot specify --{s} with action = {args.action}")
                return
        # if not config['input_time']:
        #     config['input_time'] = 30
        if not config['max_fitness']:
            config['max_fitness'] = 999999999 if config['methodology'] == "big_run" else 1
        if not config['epochs']:
            config['epochs'] = 1000
        if not config['eons_params']:
            config["eons_params"] = os.path.join(directory, 'eons', 'std.json')
        if not config['processor_params']:
            config["processor_params"] = os.path.join(directory, 'config', 'caspian.json')
        if not config['processes']:
            config['processes'] = 4
        if not config['training_network']:
            config["training_network"] = os.path.join(directory, 'networks', 'experiment_tenn1_train.json')
        if not config['logfile']:
            config["logfile"] = "tenn1_train.log"
        if not config['label']:
            config["label"] = ""
        if not config['proc_ticks']:
            config["proc_ticks"] = 10
        if not config['runs']:
            config["runs"] = 1
    else:
        illegal = [
            'eons_params', 'processor_params', 'processes', 'proc_ticks',
            'epochs', 'training_network', 'max_fitness', 'graph',
        ]
        for s in illegal:
            if config[s]:
                print(f"Cannot specify --{s} with action = {args.action}")
                return
        if config['noviz']:
            config['viz'] = False
        else:
            config['viz'] = True
        if not config['viz_delay']:
            config['viz_delay'] = 0.016
        if args.action == "test" or args.action == "run":
            # if config['car_lanes']:
            #     print("Cannot specify --car_lanes with action = test.")
            #     return
            if not config['network']:
                config["network"] = os.path.join(directory, 'networks', 'experiment_tenn1.json')
        if args.action == "test":
            if not config['testing_data']:
                config["testing_data"] = "validation.csv"

    config["environment"] = "tennbots-v00"

    # Do the appropriate action
    if args.action == "train":
        train(**config)
    elif args.action == "test":
        test(**config)
    else:
        run(**config)


if __name__ == "__main__":
    main()
