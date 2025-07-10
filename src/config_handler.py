import os
import glob
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
OPENWAKEWORD_MODELS_DIR = os.path.join('third_party', 'openwakeword', 'custom_models')
PORCUPINE_MODELS_DIR = os.path.join('third_party', 'porcupine', 'custom_models')

DEFAULT_GLOBAL = {
    'rms_filter': 50,
    'debounce_ms': 200,
    'cooldown_s': 1.5,
}
DEFAULT_MODEL = {
    'sensitivity': 0.05,
    'api_url': 'https://api.example.com/webhook',
}

def discover_models():
    models = set()
    # OpenWakeWord: .onnx files
    oww_models = glob.glob(os.path.join(OPENWAKEWORD_MODELS_DIR, '*.onnx'))
    models.update([os.path.splitext(os.path.basename(f))[0] for f in oww_models])
    # Porcupine: .ppn files
    porcupine_models = glob.glob(os.path.join(PORCUPINE_MODELS_DIR, '*.ppn'))
    models.update([os.path.splitext(os.path.basename(f))[0] for f in porcupine_models])
    return sorted(models)

class ConfigHandler:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.config = None
        self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            self.config = self._default_config()
            self.save()
        else:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        self._sync_models()

    def save(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _default_config(self):
        models = discover_models()
        return {
            'global': DEFAULT_GLOBAL.copy(),
            'models': {m: DEFAULT_MODEL.copy() for m in models}
        }

    def _sync_models(self):
        models = discover_models()
        if 'models' not in self.config:
            self.config['models'] = {}
        for m in models:
            if m not in self.config['models']:
                self.config['models'][m] = DEFAULT_MODEL.copy()
        # Remove models that no longer exist
        for m in list(self.config['models'].keys()):
            if m not in models:
                del self.config['models'][m]

    def get_global(self):
        return self.config.get('global', DEFAULT_GLOBAL.copy())

    def set_global(self, settings):
        self.config['global'] = settings
        self.save()

    def get_model(self, model_name):
        return self.config['models'].get(model_name, DEFAULT_MODEL.copy())

    def set_model(self, model_name, settings):
        self.config['models'][model_name] = settings
        self.save()

    def get_all_models(self):
        return list(self.config['models'].keys())

    def get_config(self):
        return self.config 