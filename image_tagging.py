
import os
from os.path import join, dirname
import yaml
import json
import sys
import zipfile

from watson_developer_cloud import VisualRecognitionV3


class Main(object):
    """
    A class that gives access to this test application
    """

    #The list of products obtained
    searched_products = list()
    products = None

    def __init__(self):
        #Check if we have credentials stored, if so, load them
        import pdb
        pdb.set_trace()
        if os.path.isfile('config.yml'):
            # LOGGER.info('Getting Amazon credentials from stored file...')
            self.config = yaml.safe_load(open('config.yml'))
            VISUAL_RECOGNITION_KEY = self.config['visual-recognition']['api-key']
            if VISUAL_RECOGNITION_KEY:
                # LOGGER.info('Got Amazon credentials!')
                self.visual_recognition = VisualRecognitionV3('2016-05-20', api_key=VISUAL_RECOGNITION_KEY)
            else:
                # LOGGER.error('Some mandatory Amazon credentials are empty. Aborting...')
                sys.exit()
        else:
            # LOGGER.error('Could not find amazon_credentials_file.json file. Aborting...')
            sys.exit()

    def process_images_visual_recognition(self):
        import pdb
        pdb.set_trace()
        # We can send a zip file to Visual Recognition, so let's try
        zf = zipfile.ZipFile("sample-images.zip", "w")
        for dirname, subdirs, files in os.walk('sample_images'):
            for filename in files:
                if isinstance(filename, str) and filename.endswith(('.jpg', '.png')):
                    zf.write(os.path.join(dirname, filename), arcname=filename)
        zf.close()
        import pdb
        pdb.set_trace()
        if self.visual_recognition:
            with open('sample-images.zip', 'rb') as image_file:
                results = json.dumps(self.visual_recognition.classify(images_file=image_file,
                                                                      threshold=0.1), indent=2)
            if results:
                print results

        os.remove('sample-images.zip')
if __name__ == '__main__':
    app = Main()
    app.process_images_visual_recognition()



