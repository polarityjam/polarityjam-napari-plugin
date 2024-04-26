def create_parser_napari(parser):
    from polarityjam.napari_plugin.startup import run_napari  # type: ignore
    p = parser.create_command_parser('napari', run_napari, 'Launch the Napari plugin for polarityjam.')
    return p
