import configparser
import importlib
import os

from pprint import pprint

current_dir = os.path.dirname(os.path.abspath(__file__))

def write_config(kwargs, config_path, verbose = False):
  """
  Writes .cfg file given dictionary.
  """
  if verbose: print(kwargs)
  config = configparser.RawConfigParser()

  for key, value in kwargs.items():
    value_type = type(value).__name__

    # Adds section for type if not already included
    if not config.has_section(value_type):
      config.add_section(value_type)

    config.set(value_type, key, value)

    if verbose: print(f"{key} = {value} (typeof {value_type})")

  with open(config_path, "w") as config_file:
    config.write(config_file)

def load_config(config_path, verbose = False):
  # Load .cfg file
  config = configparser.RawConfigParser()
  config.read(config_path)

  # Loads .cfg file content into dictionary and converts to declared type.
  kwargs = {}
  for section in config.sections():
    if verbose: print(f"[{section}]")
    for key, value in config.items(section):
      if (section == "bool"):
        kwargs[key] = bool(value == "True")
      elif (section == "int"):
        kwargs[key] = int(value)
      elif (section == "float"):
        kwargs[key] = float(value)
      elif (section == "eval"):
        kwargs[key] = eval(value)
      else:
        kwargs[key] = value
      if verbose: print(f"{key} = {value}")
  
  if verbose: pprint(kwargs)
  return kwargs

def initialize_model(model, results_folder = None, verbose = False):
  model_path = "model.py"
  model_config_path = "model.cfg"

  if results_folder:
    model_path = f"{results_folder}/model.py"
    model_config_path = f"{results_folder}/model.cfg"

  config = load_config(model_config_path, verbose) 

  spec = importlib.util.spec_from_file_location('model', model_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  Model = getattr(module, "Model", None)

  # Intializes specified model.
  if model == "resnet":
    Bottleneck = getattr(module, "Bottleneck", None)
    return Model(**config, ResBlock = Bottleneck, verbose = verbose)
  else:
    return Model(**config, verbose = verbose)
