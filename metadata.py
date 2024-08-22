"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""

from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
timeunit = "seconds"


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API
    
    :return: AppMetadata object holding all necessary information.
    """
    
    # first set up some basic information
    metadata = AppMetadata(
        name="Pyannotesd Wrapper",
        description="Identifies the speaker given an audio file",  # briefly describe what the purpose and features of the app
        app_license="Apache 2.0",  # short name for a software license like MIT, Apache2, GPL, etc.
        identifier="pyannotesd-wrapper",  # should be a single string without whitespaces. If you don't intent to publish this app to the CLAMS app-directory, please use a full IRI format.
        url="https://github.com/clamsproject/app-pyannotesd-wrapper.git",  # a website where the source code and full documentation of the app is hosted
        analyzer_version='',  # use this IF THIS APP IS A WRAPPER of an existing computational analysis algorithm
        # analyzer_version=[l.strip().rsplit('==')[-1] for l in open('requirements.txt').readlines() if re.match(r'^ANALYZER_NAME==', l)][0],
        analyzer_license="",  # short name for a software license
    )
    # and then add I/O specifications: an app must have at least one input and one output
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, start="start time of a speech", end="end time of a speech",
                        label='speaker label', timeUnit=timeunit).add_description(
        "Added property 'label' stores the label for speakers (e.g., SPEAKER_01)")

    metadata.add_parameter(name='huggingface_token', description='huggingface authorization token',
                           type='string')

    # CHANGE this line and make sure return the compiled `metadata` instance
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
