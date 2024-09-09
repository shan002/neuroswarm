import pathlib
import shutil
import sys
import time
import os
import yaml
from . import jsontools as jst

# typing
from typing import override


# required for training
DEFAULT_PROJECT_BASEPATH = pathlib.Path("results")
LOGFILE_NAME = "training.log"
BESTNET_NAME = "best.json"
POPULATION_FITNESS_NAME = "population_fitnesses.log"
# used in project mode
RUNINFO_NAME = "runinfo.yaml"
BACKUPNET_NAME = "previous.json"
NETWORKS_DIR_NAME = "networks"
ARTIFACTS_DIR_NAME = "artifacts"

def _NONE1(x):
    pass


def is_project_dir(path):
    return path.is_dir() and (path / RUNINFO_NAME).is_file()


def find_lastmodified_dir(basepath):
    basepath = pathlib.Path(basepath)
    projectdirs = [(child, os.path.getmtime(child)) for child in basepath.iterdir() if is_project_dir(child)]
    return max(projectdirs, key=lambda x: x[1])[0]


def check_if_writable(path):
    if not os.access(path, os.W_OK):
        msg = f"{path} could not be accessed. Check that you have permissions to write to it."
        raise PermissionError(msg)


def get_dict(obj):
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "asdict"):
        return obj.asdict()
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {i: get_dict(obj[i]) for i in range(len(obj))}

    return vars(obj)


def get_config_dict(obj):
    if hasattr(obj, "as_config_dict"):
        return obj.as_config_dict()
    return get_dict(obj)


cache = {}
def ensure_dir_exists(path: os.PathLike, parents=True, exist_ok=True, **kwargs):
    global cache
    path = pathlib.Path(path)
    fullpath = path.resolve()
    key = str(fullpath)
    if not cache.get(key, False):
        path.mkdir(parents=parents, exist_ok=exist_ok, **kwargs)
        cache[key] = path.is_dir()


class File:
    def __init__(self, path: os.PathLike):
        self.path = pathlib.Path(path)

    def write(self, s):
        with open(self.path, 'w') as f:
            f.write(s)

    def append(self, s):
        with open(self.path, 'a') as f:
            f.write(s)

    def __add__(self, s):
        self.append(s)
        return self

    def __str__(self):
        return str(self.path)


class Logger(File):
    def __init__(self, path, firstcall=None):
        super().__init__(path)
        self._initialized = False
        self.firstcall = _NONE1 if firstcall is None else firstcall

    @override
    def append(self, s):
        if not self._initialized:
            self.firstcall(self)
            self._initialized = True
        super().append(s)


class FolderlessProject:
    isproj = False

    def __init__(self, network_path_or_jsonstr, logfile_path, name=None):
        self.name = name
        self.logfile = Logger(logfile_path)
        self._network_path_or_jsonstr = network_path_or_jsonstr
        self.bestnet_file: None | File = None
        self.bestnet = dict()

    def load_bestnet(self,
        path_or_jsonstr: None | str | os.PathLike | File | dict = None,
        update_path=True,
    ) -> None:
        if path_or_jsonstr is None:
            path_or_jsonstr = self._network_path_or_jsonstr
        if isinstance(path_or_jsonstr, File):
            path_or_jsonstr = path_or_jsonstr.path

        # HACK: on most code paths, the json file will be loaded twice (json_detect, smartload)
        detected_type = jst.json_detect(path_or_jsonstr)
        if detected_type is jst.ValidJSONFilePath:
            self.bestnet = jst.smartload(path_or_jsonstr)
            self.bestnet_file = File(path_or_jsonstr)
        elif detected_type is jst.ValidJSONStr:
            self.bestnet = jst.smartload(path_or_jsonstr)
            self.bestnet_file = None
        elif detected_type is dict:
            self.bestnet = path_or_jsonstr
            self.bestnet_file = None
        else:
            raise ValueError("Unsupported type for argument 'path_or_jsonstr'")

    def check_bestnet_writable(self):
        # check that the best network path is writable
        path = pathlib.Path(self.bestnet_file.path)
        if path.is_file():
            check_if_writable(path)
            print(f"WARNING: The output network file\n    {path}\nexists and will be overwritten!")
        elif path.is_dir():
            raise OSError(1, f"{path} is a directory. Please specify a valid path for the output network file.")
        else:  # parent dir probably doesn't exist.
            if path.parent.is_dir():
                try:
                    f = open(path, 'ab')
                except BaseException as err:
                    raise err
                finally:
                    f.close()  # type: ignore
            else:
                raise OSError(2, f"One or more parent directories are missing. Cannot write to {path}.")

    def backup_bestnet(self, new_path=None):
        if self.bestnet_file.path.is_file():
            if new_path is None:
                new_path = self.bestnet_file.path.with_name(BACKUPNET_NAME)
            self.bestnet_file.path.rename(new_path)


class Project(FolderlessProject):
    isproj = True

    def __init__(self, name=None, path=None):
        self.name = name
        self.root = pathlib.Path(path) if path is not None else None
        if name is None and isinstance(path, pathlib.Path):
            # TODO: check if path contains runinfo.yaml and try to use name from that
            # try to determine name from path
            self.name = self.root
            raise NotImplementedError("TODO: determine project name from path")
        elif name is not None and path is None:
            # if name is given, use as project directory name
            self.root = pathlib.Path(DEFAULT_PROJECT_BASEPATH / name)
        self.bestnet_file = File(self.root / BESTNET_NAME)
        self.networks = Networks(self, NETWORKS_DIR_NAME)
        self.logfile = Logger(self.root / LOGFILE_NAME)
        self.popfit_file = Logger(self.root / POPULATION_FITNESS_NAME)
        self.popfit_file.firstcall = self._default_firstcall

    def load_bestnet(self,
        path_or_jsonstr: None | str | os.PathLike | File | dict = None,
        update_path=True,
    ) -> None:
        if not path_or_jsonstr:
            path_or_jsonstr = self.bestnet_file
        super().load_bestnet(path_or_jsonstr, update_path)

    def _default_firstcall(self, f):
        f.write(f"{time.time()}\t{0}\t[]\n")

    def make_root_interactive(self):
        create_parents = False
        if self.root.is_dir():
            s = input(f"Project folder already exists:\n\t{str(self.root)}\n'y' to continue, 'rm' to delete the contents of the folder, anything else to exit.")  # noqa: E501
            if s.lower() not in ('y', 'yes', 'rm'):
                print("Exiting. Your filesystem has not been modified.")
                sys.exit(1)
            if s.lower() == 'rm':
                shutil.rmtree(self.root)   # type: ignore[reportArgumentType]
                print(f"Deleted {self.root}.")
        elif not self.root.parent.is_dir():
            print("WARNING: You're trying to put the project in")
            print(str(self.root.parent))
            print("but some part of it does not exist! Would you like to create it?")
            s = input("Type 'y' to create it, anything else to exit.")
            if s.lower() not in ('y', 'yes'):
                print("Exiting. Your filesystem has not been modified.")
                sys.exit(1)
            else:
                create_parents = True
        print(f"Creating project folder at {self.root}")
        ensure_dir_exists(self.root, parents=create_parents)

    @property
    @override
    def logfile_path(self):
        return self.root / LOGFILE_NAME

    @property
    def runinfo_path(self):
        return self.root / RUNINFO_NAME

    @property
    def popfit_path(self):
        return self.root / POPULATION_FITNESS_NAME

    def log_popfit(self, info):
        self.popfit_file += f"{time.time()}\t{info.i}\t{repr(info.fitnesses)}\n"

    def ensure_dir(self, relpath, parents=True, exist_ok=True, **kwargs):
        path = self.root / relpath
        ensure_dir_exists(path, parents=parents, exist_ok=exist_ok, **kwargs)
        return path

    def ensure_file_parents(self, relpath, parents=True, exist_ok=True, **kwargs):
        path = self.root / relpath
        ensure_dir_exists(path.parent, parents=parents, exist_ok=exist_ok, **kwargs)
        return path

    def save_yaml_artifact(self, name, obj):
        artifacts = pathlib.Path(ARTIFACTS_DIR_NAME)
        with open(self.ensure_file_parents(artifacts / name), "w") as f:
            yaml.dump(get_config_dict(obj), f)


class Networks:
    def __init__(self, project, relpath):
        self.path = project.root / relpath

    def get_file(self, epoch, population_id):
        ensure_dir_exists(self.path, parents=False)
        return File(self.path / f"e{epoch}-{population_id}.json")


