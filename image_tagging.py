
import os
import yaml
import json
import zipfile
import pandas
import simplejson
import ntpath

from watson_developer_cloud import VisualRecognitionV3
from clarifai.client import ClarifaiApi


class ImageTagger(object):
    """
    A class that gives access to this test application
    """

    VISUAL_RECOGNITION_KEY = ''
    CLARIFAI_CLIENT_ID = ''
    CLARIFAI_CLIENT_SECRET = ''

    def __init__(self):
        self.data_frame = pandas.DataFrame()
        self.images_names = list()
        self.apis = ['VisualRecognition', 'Clarifai']
        self.visual_recognition = None
        self.clarifai = None
        self.configured = False

    def configure_tagger(self, config_file):
        """
        Reads the API credentials from the specified YAML file and initializes API clients
        :param config_file: The file path to the config YAML file
        :return: True if config file was parsed and API clients initialized correctly
        """
        #Check if provided config yaml file actually does exist
        if os.path.isfile(config_file):
            config = yaml.safe_load(open(config_file))
            # Get config data
            self.VISUAL_RECOGNITION_KEY = config['visual-recognition']['api-key']
            self.CLARIFAI_CLIENT_ID = config['clarifai']['client-id']
            self.CLARIFAI_CLIENT_SECRET = config['clarifai']['client-secret']
            if self.VISUAL_RECOGNITION_KEY:
                self.visual_recognition = VisualRecognitionV3('2016-05-20', api_key=self.VISUAL_RECOGNITION_KEY)
            if self.CLARIFAI_CLIENT_ID and self.CLARIFAI_CLIENT_SECRET:
                self.clarifai = ClarifaiApi(app_id=self.CLARIFAI_CLIENT_ID, app_secret=self.CLARIFAI_CLIENT_SECRET)
            if self.visual_recognition and self.clarifai:
                self.configured = True

        else:
            print('Could not find config file')
            return False

    def process_images_visual_recognition(self, folder_name=None, store_results=False):
        """
        Processes the specified image folder using the Visual Recognition API
        :param folder_name: The complete path where the images are
        :param store_results: Indicates if obtained response should be stored as JSON file
        :return: A DataFrame containing the available data
        """
        if folder_name:
            # We can send a zip file to Visual Recognition, so let's try
            zf = zipfile.ZipFile("sample-images.zip", "w")
            for dirname, subdirs, files in os.walk(folder_name):
                for filename in files:
                    if isinstance(filename, str) and filename.endswith(('.jpg', '.png')):
                        self.images_names.append(filename)
                        zf.write(os.path.join(dirname, filename), arcname=filename)
            zf.close()
            # Check we have the client API instance
            if self.visual_recognition:
                with open('sample-images.zip', 'rb') as image_file:
                    results = simplejson.dumps(self.visual_recognition.classify(images_file=image_file,
                                                                                threshold=0.1),
                                               indent=4,
                                               skipkeys=True,
                                               sort_keys=True)
                    if store_results:
                        fd = open('visual_recognition_classifications_results.json', 'w')
                        fd.write(results)
                        fd.close()

                # Generate a dict with a list of tuples with all tags found per image
                vr_results = dict()
                try:
                    vr_data = json.loads(results.decode('string-escape').strip('"'))
                    if 'images' in vr_data.keys():
                        # print(vr_data)
                        for image in vr_data['images']:
                            tags_found = []
                            if 'image' in image.keys():
                                image_name = self.path_leaf(image['image'])
                                if 'classifiers' in image.keys():
                                    for tag in image['classifiers'][0]['classes']:
                                        if 'class' and 'score' in tag.keys():
                                            tag_found = (tag['class'], tag['score'])
                                            tags_found.append(tag_found)
                                    vr_results[image_name] = tags_found
                except Exception as ex:
                    print 'COULD NOT LOAD:', ex
                data_series = pandas.Series(vr_results, index=self.images_names, name='VisualRecognition')
                data_frame = pandas.DataFrame(data_series, index=self.images_names, columns=self.apis)

            # Removed generated zip file
            os.remove('sample-images.zip')
            return data_frame
        else:
            return None

    def process_images_clarifai(self, folder_name=None):
        """
        Processes the specified image folder using the Clarifai API
        :param folder_name: The complete path where the images are
        :return: A DataFrame containing the available data
        """
        if folder_name:
            # Check we have the client API instance
            if self.clarifai:
                # Generate a dict with a list of tuples with all tags found per image
                clarifai_results = dict()
                try:
                    open_files = []
                    # Generate a dict with a list of tuples with all tags found per image
                    clarifai_results = dict()
                    # Build the list of open files to be sent to clarifai
                    for dirname, subdirs, files in os.walk(folder_name):
                        for filename in files:
                            if isinstance(filename, str) and filename.endswith(('.jpg', '.png')):
                                image_file = open(os.path.join(dirname, filename), 'rb')
                                image = (image_file, filename)
                                self.images_names.append(filename)
                                open_files.append(image)
                                # Call Clarifai API
                                clarifai_data = self.clarifai.tag_images(open_files)

                                if 'results' in clarifai_data.keys():
                                    # print(vr_data)
                                    for image in clarifai_data['results']:
                                        tags_found = []
                                        if 'result' in image.keys():
                                            image_name = filename
                                            # Try to get the tags obtained
                                            result = image['result']
                                            if result:
                                                if 'tag' in result.keys():
                                                    tags = image['result']['tag']['classes']
                                                    list_tags = []
                                                    probs = image['result']['tag']['probs']
                                                    if tags and probs:
                                                        list_tags = zip(tags, probs)
                                                        clarifai_results[image_name] = list_tags
                except Exception as ex:
                    print 'COULD NOT LOAD:', ex
                data_series = pandas.Series(clarifai_results, index=self.images_names, name='Clarifai')
                data_frame = pandas.DataFrame(data_series, index=self.images_names, columns=self.apis)
                return data_frame

        else:
            return None

    def use_all(self, folder):
        """
        :param folder: The folder containing images to be tagged
        A wrapper that will use all available APIs
        :return: A DataFrame with all available data
        """
        results = None
        if self.configured and os.path.isdir(folder):
            vr_df = self.process_images_visual_recognition(folder)
            cr_df = self.process_images_clarifai(folder)
            # Merge both dataframes into one
            results = pandas.concat(vr_df, cr_df)

        return results

    def path_leaf(self, path):
        """
        A simple helper function that returns the last path (the file) of a path
        :param path: The whole path
        :return: The filename
        """
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


if __name__ == '__main__':
    app = ImageTagger()
    #app.process_images_visual_recognition()
    app.process_images_clarifai(folder_name='sample_images')



