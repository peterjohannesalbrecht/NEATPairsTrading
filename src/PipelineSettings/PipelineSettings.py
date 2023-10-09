import datetime
import json


class PipelineSettings:
    """Handles the settings for the pipeline.

    Is used to load and provide predefined settings by
    the user during the run of the pipeline.
    """

    def __init__(self) -> None:
        self.settings_dict = None

    def load_settings(self) -> dict:
        """Load the predefined settings from a .json-file."""
        with open('src/Input/pipeline_settings.json') as file:
            self.settings_dict = json.load(file)
        self.settings_dict['starting_time'] = datetime.datetime.now().strftime(
            '%Y-%m-%d_%H:%M:%S'
        )
        return self.settings_dict
