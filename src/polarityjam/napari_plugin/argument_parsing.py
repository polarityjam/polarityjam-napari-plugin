from polarityjam.napari_plugin.startup import run_napari
def create_parser_napari(parser):
    p = parser.create_command_parser('napari', run_napari, 'Launch the Napari plugin for polarityjam.')
    return p