import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from io import BytesIO
import caspian
from tqdm import tqdm
from copy import deepcopy

from common.experiment import TennExperiment, train, get_parsers as common_get_parsers
from common.utils import make_template
from common import env_tools as envt
from common.argparse import ArgumentError

from rss.CaspianBinaryController import CaspianBinaryController
from rss.CaspianBinaryRemappedController import CaspianBinaryRemappedController
from rss.gui import TennlabGUI
import rss.graphing as graphing
from swarmsim.world.RectangularWorld import RectangularWorld

from RunnerController import RunnerController
from swarmsim.config import register_dictlike_type, get_agent_class
from swarmsim.world.RectangularWorld import RectangularWorldConfig
from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber
from swarmsim.world.simulate import main as simulator
from swarmsim import metrics
from CatchRunnerMetric import CatchRunnerMetric
import matplotlib.pyplot as plt

class HunterVsRunnerExperiment(TennExperiment):
    def __init__(self, args):
        if not os.path.isabs(args.eons_params):
            args.eons_params = os.path.join(parent_dir, args.eons_params)
        if not os.path.isabs(args.snn_params):
            args.snn_params = os.path.join(parent_dir, args.snn_params)
        super().__init__(args)
        self.world_yaml = args.world_yaml
        self.run_info = None
        self.n_inputs, self.n_outputs, _, _ = CaspianBinaryController.get_default_encoders()
        self.track_history = getattr(args, "track_history", False) or getattr(args, "log_trajectories", False)
        self.log_trajectories = getattr(args, "log_trajectories", False)
        self.start_paused = getattr(args, "start_paused", False)
        # self.runner_position = None
        # self.runner_position = getattr(args, "runner_position", (7, 8))
        self.trials = 10
        rng = np.random.RandomState(args.trial_seed)
        self.trial_seeds = rng.randint(low=0, high=np.iinfo(np.uint32).max, size=self.trials).tolist()
        self.log("Initialized Hunter vs RunnerExperiment")
        self.log(f"Using trial_seed={args.trial_seed} → trial_seeds={self.trial_seeds}")

    def get_rand_pos_within_region(self, region):
        region = np.array(region)
        mins = region.min(axis=0)
        maxs = region.max(axis=0)
        point = np.random.uniform(low=mins, high=maxs)

        return point
        
    def get_rand_pos_outside_goal(self, config):
        goal = [object for object in config.objects if object['name'] == 'goal'][0]
        agent_radius = config.spawners[0]['agent']['agent_radius']
        CLEARANCE_FROM_GOAL = 0.1
        min_distance_from_goal = goal['agent_radius'] + agent_radius + CLEARANCE_FROM_GOAL
        world_width, world_height = config.size
        margin = 1

        # Run the loop until get a point outside the goal
        while True:
            candidate = np.array([
                np.random.uniform(margin, world_width - margin),
                np.random.uniform(margin, world_height - margin)
            ])
            if np.linalg.norm(candidate - goal['position']) > min_distance_from_goal:
                break

        return tuple(candidate)


    def simulate(self, processor, network, init_callback=lambda config: config):
        network.set_data("processor", self.processor_params)

        # Register controller types
        register_dictlike_type('controller', "CaspianBinaryController", CaspianBinaryController)
        register_dictlike_type('controller', "CaspianBinaryRemappedController", CaspianBinaryRemappedController)
        register_dictlike_type('controller', "RunnerController", RunnerController)

        # Load the world configuration from YAML
        config = RectangularWorldConfig.from_yaml(self.world_yaml)
        config.stop_at = self.cycles
        config.meta = {}

        # Load experiment-specific parameters from the YAML
        exp_params = getattr(config, "experiment", {})
        runner_speed = exp_params.get("runner_speed", 0.1)
        runner_color = exp_params.get("runner_color", [0, 255, 0])
        runner_region = exp_params.get("runner_region", None)
        window_size = exp_params.get("window_size", [300, 300])
        agent_config = config.spawners[0]['agent']

        # Set random goal position within the goal region
        goal = [object for object in config.objects if object['name'] == 'goal'][0]
        goal['position'] = self.get_rand_pos_within_region(goal['region'])

        # Override number of hunter agents if -N is passed
        n_agents = getattr(self.args, 'agents', None)
        if n_agents is not None:
            config.spawners[0]['n'] = n_agents
            self.log(f"Overriding number of agents to {n_agents}")

        agent_config['track_io'] = self.track_history
        controller_config = agent_config['controller']
        controller_config['network'] = network

        # Create a standalone runner agent by copying the spawner’s base agent config
        base_agent_config = deepcopy(agent_config)

        # Remove any network key from the controller config
        if "network" in base_agent_config.get("controller", {}):
            del base_agent_config["controller"]["network"]

        agent_cls, runner_config = get_agent_class(base_agent_config)

        # runner_config.position = self.runner_position
        runner_config.team = "runner"  # tag as runner
        runner_config.body_color = runner_color
        if runner_region is not None:
            runner_config.position = self.get_rand_pos_within_region(runner_region)

        # self.runner_position = self.get_rand_pos_outside_goal(config)

        # Override the controller to use RunnerController.
        runner_config.controller = {"type": "RunnerController", "speed": runner_speed}

        # Add the runner agent to the standalone agents list
        if not hasattr(config, "agents"):
            config.agents = []
        config.agents.append(runner_config)

        config.metrics = [CatchRunnerMetric()]

        def check_stop(world):
            output =  world.metrics[0].out_current()[1]
            return output is not None

        def callback(world, screen):
            a = world.selected
            if a and self.iostream:
                self.iostream.write_json({
                    "Neuron Alias": a.neuron_ids,
                    "Event Counts": a.neuron_counts
                })

        gui = TennlabGUI(x=0, y=0, h=window_size[1], w=window_size[0])
        gui.position = "sidebar_right"
        if not getattr(self, "viz", False):
            gui = False

        world_subscriber = WorldSubscriber(func=callback)

        # allow for callback to modify config
        config = init_callback(config)

        world = simulator(
            world_config=config,
            subscribers=[world_subscriber],
            gui=gui,
            show_gui=bool(gui),
            start_paused=self.start_paused,
            stop_detection=check_stop
        )

        return world


    def extract_fitness(self, world_output: RectangularWorld):
        return world_output.metrics[0].out_current()[1] if world_output.metrics else 0.0


    def fitness(self, processor, network, init_callback=lambda config: config):
        fitnesses = []
        # print(f"During training: {self.trial_seeds}")
        for i, seed in enumerate(self.trial_seeds):
            np.random.seed(seed)

            # print(f"trial - {i+1}: fitness = {fitness}")
            # print(f"[Trial {i+1}] Runner position: {self.runner_position}"
            world_final_state = self.simulate(processor, network, init_callback)
            f = self.extract_fitness(world_final_state)
            if f is not None:
                fitnesses.append(f)

        avg = float(np.mean(fitnesses)) if fitnesses else 0.0
        print(f"Fitnesses: {fitnesses} → mean = {avg}")
        return avg

    def as_config_dict(self):
        d = super().as_config_dict()
        d.update({"world_yaml_path": self.world_yaml})
        return d

    def save_network(self, net, path):
        if 'encoder_ticks' not in self.app_params:
            world = self.get_sample_world(delete_rss=False)
            hunter = next(agent for agent in world.population if getattr(agent, "team", None) != "runner")
            self.app_params.update({'encoder_ticks': hunter.controller.neuro_tpc})
        super().save_network(net, path)


    def get_sample_world(self, delete_rss=True):
        cycles = self.cycles
        self.cycles = 0
        proc = caspian.Processor(self.processor_params)
        template_net = make_template(proc, self.n_inputs, self.n_outputs)
        world = self.simulate(proc, template_net)
        self.cycles = cycles
        if delete_rss:
            self.delete_rss()
        return world

    def delete_rss(self):
        if 'rss' in globals():
            del rss

    def save_artifacts(self, evolver, *args, **kwargs):
        if super().save_artifacts(evolver, *args, **kwargs) is None:
            return
        self.p.save_yaml_artifact("env.yaml", self.get_sample_world(delete_rss=False))
        self.delete_rss()

    def get_env_info(self):
        d = super().get_env_info()
        try:
            swarmsim_path = envt.module_editable_path('swarmsim')
            d['.dependencies'].update({
                'swarmsim': {
                    'path': str(swarmsim_path.resolve()),
                    "branch": envt.get_branch_name(swarmsim_path),
                    "HEAD": envt.git_hash(swarmsim_path),
                    "status": [s.strip() for s in envt.git_porcelain(swarmsim_path).split('\n')],
                    'version': envt.get_module_version('swarmsim'),
                },
            })
        except Exception:
            d['.dependencies'].update({'swarmsim': envt.get_module_version('swarmsim')})
        return d

def run(app, args):
    if args.stdin == "stdin":
        proc = None
        net = None
    else:
        proc = caspian.Processor(app.processor_params)

        if app.net is None:
            from common.utils import make_template
            app.net = make_template(proc, app.n_inputs, app.n_outputs)
        net = app.net

    fitnesses = []
    overall_fits = []
    rng = np.random.RandomState(args.trial_seed)
    trial_seeds = rng.randint(low=0, high=np.iinfo(np.uint32).max, size=args.trials).tolist()
    # print(f"During running: {trial_seeds}")
    for i, seed in enumerate(trial_seeds):
        np.random.seed(seed)

        world = app.simulate(proc, net)
        f     = app.extract_fitness(world)
        fitnesses.append(f)

        avg = sum(fitnesses) / len(fitnesses)
        overall_fits.append(avg)
        print(f"[run] trial {i+1}/{args.trials}: {f:6.4f} | Overall Fitness: {avg:6.4f}")

    final = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    print(f"\nFitness after {args.trials} trials: {final:8.4f}")

    if getattr(args, 'plot_fit', False):
        plt.figure()
        plt.plot(range(1, len(overall_fits)+1), overall_fits, marker='o')
        plt.xlabel('Trial #')
        plt.ylabel('Fitness')
        plt.title('Fitness over Trials')
        plt.grid(True)
        plt.tight_layout()
        plt.ylim(0, 1.1)
        plt.show()


    if args.log_trajectories:
        graphing.plot_multiple(world)
        graphing.export(world, output_file=app.p.ensure_file_parents("agent_trajectories.xlsx"))
    return final


def test(app, args):
    proc = caspian.Processor(app.processor_params)
    net = app.net
    if args.positions:
        from rss.rss2 import PredefinedInitialization, SCALE
        import pandas as pd
        with open(args.positions, 'rb') as f:
            xlsx = f.read()
        xlsx = pd.ExcelFile(BytesIO(xlsx))
        sheets = xlsx.sheet_names
        n_runs = len(sheets)
        pinit = PredefinedInitialization()
        def setup_i(i):
            pinit.set_states_from_xlsx(args.positions, sheet_number=i)
            pinit.rescale(SCALE)
            def setup(world):
                world.init_type = pinit
                return world
            return setup
        fitnesses = [app.fitness(proc, net, setup_i(i)) for i in tqdm(range(n_runs))]
        # print(fitnesses)
        return fitnesses
    else:
        raise ArgumentError(args.positions, "Positions not specified")

def get_parsers(parser, subpar):
    sp = subpar.parsers
    for sub in sp.values():
        sub.add_argument('-N', '--agents', type=int, help="# of agents to run with.", default=None)
        sub.add_argument('--world_yaml', default="./world.yaml",
                         type=str, help="path to yaml config for sim")
        sub.add_argument('--trial_seed', '-S', type=int, default=12345, help="[train] seed for generating 10 trial-seeds")
        sub.add_argument('-T','--trials', type=int, default=1, help="number of independent runs to average over")
    sp['train'].add_argument('--label', help="[train] label to put into network JSON (key = label).")
    sp['run'].add_argument('--track_history', action='store_true',
                           help="pass this to enable sensor vs. output plotting.")
    sp['run'].add_argument('--plot_fit', action='store_true',
                           help="after running all trials, plot fitness vs. trial number")
    sp['run'].add_argument('--log_trajectories', action='store_true',
                           help="pass this to log sensor vs. output to file.")
    sp['run'].add_argument('--start_paused', action='store_true',
                           help="pass this to pause the simulation at startup. Press Space to unpause.")
    sp['test'].add_argument('--positions', default=None, help="file containing agent positions")
    sp['test'].add_argument('-p', '--processes', type=int, default=1,
                           help="number of threads for concurrent fitness evaluation.")
    return parser, subpar

def main():
    parser, subpar = common_get_parsers()
    parser, subpar = get_parsers(parser, subpar)
    args = parser.parse_args()
    args.environment = getattr(args, "environment", "connorsim_snn_eons-v01")
    if getattr(args, "project", None) is None and getattr(args, "logfile", None) is None:
        args.logfile = "tenn2_train.log"
    app = HunterVsRunnerExperiment(args)
    if args.action == "train":
        train(app, args)
    elif args.action == "test":
        test(app, args)
    elif args.action == "run":
        run(app, args)
    else:
        raise RuntimeError("No action selected")

if __name__ == "__main__":
    main()

# python hunter_vs_runner.py run --root ../../results_sim/hopper/250425/farp/6 --cy 2000 --trials 10 --trial_seed 42 --plot_fit
# python hunter_vs_runner.py train --root out/ --save_best -p 48 -T 10 --cy 2000 --epochs 500 --trial_seed 410 --eons_seed 20
