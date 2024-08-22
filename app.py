"""
DELETE THIS MODULE STRING AND REPLACE IT WITH A DESCRIPTION OF YOUR APP.

app.py Template

The app.py script does several things:
- import the necessary code
- create a subclass of ClamsApp that defines the metadata and provides a method to run the wrapped NLP tool
- provide a way to run the code as a RESTful Flask service


"""

import argparse
import logging

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes, DocumentTypes
import torch
from pyannote.audio import Pipeline
import ffmpeg
import metadata


class PyannotesdWrapper(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif)

        # Try to get Audio files
        docs = mmif.get_documents_by_type(DocumentTypes.AudioDocument)

        if docs:
            # if the input is a audio file
            new_view = self._new_view(parameters)
            for doc in docs:
                location_path = doc.location_path(nonexist_ok=False)
                self._speaker_diarizer(location_path, new_view, parameters['huggingface_token'])
        else:
            # if the input is a video file
            docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
            new_view = self._new_view(parameters)
            for doc in docs:
                video_location_path = doc.location_path(nonexist_ok=False)
                # pyannote only takes audio files (preferably .wav format), so we need to convert the video into
                # a wav file and store it locally before we send it to pyannote
                wav_location_path = video_location_path.replace(".mp4", ".wav")
                # ffmpeg converts the video into a wav file with channel 1 and sample rate 16000
                ffmpeg.input(video_location_path).output(wav_location_path, ac=1, ar=16000).run()
                # send the converted wav file to pyannote
                self._speaker_diarizer(wav_location_path, new_view, parameters['huggingface_token'])

        return self.mmif

    def _new_view(self, runtime_config):
        view = self.mmif.new_view()
        view.metadata.app = self.metadata.identifier
        self.sign_view(view, runtime_config)
        view.new_contain(AnnotationTypes.TimeFrame)
        return view

    def _speaker_diarizer(self, file, new_view, huggingface_token):
        """
        Run the pyannote speaker diarization tool to the given file (audio or video), using the HuggingFace token
        given by the user.
        """
        # load pretrained pipeline
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',
                                            use_auth_token=huggingface_token)
        # send pipeline to GPU (when available)
        if torch.cuda.is_available():
            pipeline.to(torch.device('cuda'))

        diarization = pipeline(file)

        # write the diarization results into TimeFrames
        for segment, track, label in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            new_view.new_annotation(AnnotationTypes.TimeFrame, frameType="speech", start=start_time, end=end_time,
                                    label=label, timeUnit=metadata.timeunit)


def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    return PyannotesdWrapper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
