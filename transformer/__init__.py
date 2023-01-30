try:
    import transformer.modules as modules
    import posEncoders
    import baseT
    
except ImportError:
    import transformer.modules as modules
    import .posEncoders
    import .baseT