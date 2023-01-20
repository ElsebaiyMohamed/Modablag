try:
    import attentions
    import posEncoders
    import encoders
except ImportError:
    from .attentions import *
    from .posEncoders import *
    from .encoders import *