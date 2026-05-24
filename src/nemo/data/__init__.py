"""
NEMO datainnhentingsmodul.

Eksporterer pipeline-inngangspunkter og transformasjonsverktøy.
Importer enkeltmoduler direkte for å unngå sirkularitet ved modul-kjøring:
    from nemo.data.pipeline import kjor_pipeline
"""

# Lazy imports — hindrer sirkularitetsproblemer ved `python -m nemo.data.pipeline`
__all__ = [
    "hp_filter",
    "log_diff",
    "kjor_pipeline",
    "OBSERVASJONSVARIABLER",
]


def __getattr__(name: str):
    if name in __all__:
        from nemo.data import pipeline as _pipeline
        return getattr(_pipeline, name)
    raise AttributeError(f"Modul 'nemo.data' har ikke attributt '{name}'")
