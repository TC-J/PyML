import inspect

def filter_keys(keys: list[str], d: dict) -> dict:
    return {key: d[key] for key in keys if key in d}


def fn_sig_keys(fn_sig) -> list[str]:
    return inspect.signature(fn_sig).parameters.keys()


def get_fn_sig_kwargs(fn_sig, kwargs: dict):
    return filter_keys(keys=fn_sig_keys(fn_sig), d=kwargs)