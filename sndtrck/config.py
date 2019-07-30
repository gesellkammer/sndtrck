from notifydict import NotifyDict
import appdirs
import os
import json
import logging
import sys


logger = logging.getLogger("sndtrck")


class ConfigDict(NotifyDict):
    def __init__(self, allowedkeys):
        self._allowedkeys = allowedkeys
        NotifyDict.__init__(self)

    def __setitem__(self, key, value):
        if key in self._allowedkeys:
            isvalid, errormsg = _config_value_isvalid(key, value)
            if isvalid:
                NotifyDict.__setitem__(self, key, value)
            else:
                raise ValueError(errormsg)
        else:
            raise KeyError(f"Unknown key: {key}")

    def set(self, key, value):
        return self.__setitem__(key, value)

    def override(self, value, key):
        """
        If value is not None, return value directly
        Otherwise, return config[value]

        key must be present in the dict

        This operation is the opposite of .get, in the sense that
        the value given is queried before the value stored
        in this config (config is not modified)
        """
        if value is not None:
            return value
        return self.get(key, default)


DEFAULT_CONFIG = {
    'spectrum.allow_negative_times': False,
    'spectrum.minbreakpoints': 1,
    'spearformat': 'sdif',
    'spectrum.show.method': 'builtin',
    'show.show_chord.method': 'musicxml.png',
    'breakpointgap.percentile': 25,
    'breakpointgap.partial_percentile': 25,
    'csound.realtime.beadsyntflags': 0,
    'csound.render.beadsyntflags': 7,
    'csound.backend.win': 'portaudio',
    'csound.backend.linux': 'jack',
    'csound.backend.macos': 'coreaudio',
    'plot.backend': 'pyqtgraph',
    'render.samplerate': 44100,
    'render.method': 'loris',
    'spectrumeditor.updaterate': 12,
    'sinesynth.porttime': 0.05,
    'spectrumeditor.downsample': 2,
    'spectrumeditor.alpha': 255,
    'spectrumeditor.linewidth': 2,
    'spectrumeditor.exp': 1.0,
    'spectrumeditor.numcolors': 64,
    'spectrumeditor.gain': 1,
    'spectrumeditor.noisegain': 1,
    'spectrumeditor.numloudest': 4,
    'plot.downsample': 1,
    'plot.linewidth': 2,
    'plot.showpoints_disables_downsample': True,
    'A4': 442
}


_CONFIG_VALIDATOR = {
    'spearformat::choices': ('sdif', 'txt'),
    'spectrum.show.method::choices': ('builtin', 'spear'),
    'show.show_chord.method::choices':
        ('musicxml.png', 'musicxml', 'lily.png', 'lily.pdf'),
    'render.method::choices': ('loris', 'csound'),
    'render.samplerate::type': int,
    'plot.backend::choices': ('pyqtgraph', 'pyplot')
}


def config_getpath():
    configbasedir = appdirs.user_config_dir()
    app = 'sndtrck'
    configname = 'config.json'
    configdir = os.path.join(configbasedir, app)
    if not os.path.exists(configdir):
        logger.debug(f"Creating config directory: {configdir}")
        os.mkdir(configdir)
    return os.path.join(configdir,configname)


def config_getchoices(key):
    """
    Return a seq. of possible values for key `k`
    or None
    """
    choices = _CONFIG_VALIDATOR.get(key + "::choices", None)
    if choices is not None:
        return choices
    return None


def config_gettype(key):
    default = DEFAULT_CONFIG.get(key)
    if default is None:
        raise KeyError("Key is not present in default config")
    return type(default)
    

def _wrapdict(d, saveit=True):
    path = config_getpath()
    
    def saveconfig(d, path=path):
        logger.debug(f"Saving config to {path}")
        logger.debug("Config: %s" % json.dumps(d, indent=True))
        f = open(path, "w")
        json.dump(d, f, indent=True)

    d2 = ConfigDict(allowedkeys=DEFAULT_CONFIG.keys())      
    d2.update(d)
    d2.register(lambda *args:saveconfig(d2))
    if saveit:
        saveconfig(d)
    return d2


def _config_isvalid(d):
    for k, v in d.items():
        isvalid, errorstr = _config_value_isvalid(k, v)
        if not isvalid:
            return False, errorstr
    return True, None


def _config_value_isvalid(key, value):
    """
    Returns isvalid, errormsg

    where:
        isvalid: is True if value is a possible value for key
        errormsg: None if value is valid, an error string otherwise
    """
    choices = config_getchoices(key)
    if choices is not None:
        if value not in choices:
            return False, f"key should be one of {choices}, got {value}"
    valuetype = config_gettype(key)
    if valuetype is not None:
        if not isinstance(value, valuetype):
            vtype = type(value)
            return False, f"Expected type {valuetype} for {key}, got {vtype}"
    return True, None


def _read_config():
    configpath = config_getpath()
    d = DEFAULT_CONFIG
    if not os.path.exists(configpath):
        logger.debug("Using default config")
        configdict = d.copy()
    else:
        logger.debug(f"Reading config from disk: {configpath}")
        try:
            configdict = json.load(open(configpath))
        except:
            error = sys.exc_info()[0]
            logger.error(f"Could not read config {configpath}: {error}")
            logger.debug("Using default config")
            configdict = d.copy()
    isvalid, errorstr = _config_isvalid(configdict)
    if not isvalid:
        logger.error(f"Could not validate config: \n {errorstr}")
        c, d = configdict, d
        if c.keys() == d.keys() and all(c[k] == d[k] for k in c.keys()):
            raise RuntimeError("The default config is not valid! \n\nDEFAULT_CONFIG: \n%s" 
                               % str(d))

    if d.keys() - configdict.keys():
        logger.warning("DEFAULT_CONFIG has keys not present in read config."
                       "Consider calling resetconfig()")

    return _wrapdict(configdict)
    

_CONFIG = None


def getconfig():
    """
    Return a dictionary with all configuration options. This is a 
    persistent dictionary, in the sense that changes done to it
    are saved and retrieved in a future session. To reset the
    config to default values, call resetconfig()
    """
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _read_config()
    return _CONFIG


def resetconfig():
    path = config_getpath()
    if os.path.exists(path):
        logger.debug(f"Removing saved config at: {path}")
        os.remove(path)
    global _CONFIG
    _CONFIG = _wrapdict(DEFAULT_CONFIG)


config = getconfig()