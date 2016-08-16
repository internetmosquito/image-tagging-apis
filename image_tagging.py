
import os
import yaml
import json
import zipfile
import pandas
import simplejson
import ntpath
import base64
import time

from httplib2 import HttpLib2Error

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from oauth2client.client import GoogleCredentials

from watson_developer_cloud import VisualRecognitionV3
from clarifai.client import ClarifaiApi


class ImageTagger(object):
    """
    A class that gives access to this test application
    """

    # For testing, we use CircleCI and have ciphered the config.yml at root with this command
    # openssl aes-256-cbc -e -in config.yml -out config-cipher -k $KEY
    # Where $KEY is the value of an environment variable that must be set in your CircleCI
    VISUAL_RECOGNITION_KEY = ''
    CLARIFAI_CLIENT_ID = ''
    CLARIFAI_CLIENT_SECRET = ''
    GOOGLE_VISION_DISCOVERY_URL='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
    IMAGE_FILE_TYPES = ['png', 'jpg', 'jpeg', 'gif']

    def __init__(self):
        self.data_frame = pandas.DataFrame()
        self.images_names = list()
        self.apis = ['VisualRecognition', 'Clarifai', 'GoogleVision']
        self.visual_recognition = None
        self.clarifai = None
        self.google_vision_service = None
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
            self.GOOGLE_VISION_SECRET = config['google-vision']['api-key']
            if self.VISUAL_RECOGNITION_KEY:
                self.visual_recognition = VisualRecognitionV3('2016-05-20', api_key=self.VISUAL_RECOGNITION_KEY)
            if self.CLARIFAI_CLIENT_ID and self.CLARIFAI_CLIENT_SECRET:
                self.clarifai = ClarifaiApi(app_id=self.CLARIFAI_CLIENT_ID, app_secret=self.CLARIFAI_CLIENT_SECRET)
            if self.GOOGLE_VISION_SECRET:
                self.google_vision_service = discovery.build('vision',
                                                             'v1',
                                                             developerKey=self.GOOGLE_VISION_SECRET,
                                                             discoveryServiceUrl=self.GOOGLE_VISION_DISCOVERY_URL
                                                             )

            if self.visual_recognition and self.clarifai and self.google_vision_service:
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
        data_frame = None
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

    def process_images_clarifai(self, folder_name=None):
        """
        Processes the specified image folder using the Clarifai API
        :param folder_name: The complete path where the images are
        :return: A DataFrame containing the available data
        """
        data_frame = None
        # Check we have the client API instance
        if self.clarifai:
            # Generate a dict with a list of tuples with all tags found per image
            clarifai_results = dict()
            try:
                open_files = []
                # Generate a dict with a list of tuples with all tags found per image
                clarifai_results = dict()

                if os.path.isdir(folder_name):
                    # Get the images if the exist and if they are in the supported types
                    images = [filename for filename in os.listdir(folder_name)
                              if os.path.isfile(os.path.join(folder_name, filename)) and
                              filename.split('.')[-1].lower() in self.IMAGE_FILE_TYPES]
                    for iterator, image_file in enumerate(images):
                        image_path = os.path.join(folder_name, image_file)
                        self.images_names.append(self.path_leaf(image_path))
                        image_file = open(image_path, 'rb')
                        image = (image_file, self.path_leaf(image_path))
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
                print ('COULD NOT LOAD, reason {0}'.format(str(ex)))
            data_series = pandas.Series(clarifai_results, index=self.images_names, name='Clarifai')
            data_frame = pandas.DataFrame(data_series, index=self.images_names, columns=self.apis)
            return data_frame

    def process_images_google_vision(self, folder_name=None):
        """
        Iterates over the specified folder and returns the combined response from calling Google Cloud Vision API
        using only LABEL detection and 5 maximum per image. Since it looks like Google does not like sending more
        than 10 images per Request, we have to make sure we process every batch of 10 images until finished
        :param folder_name: The full path to the folder with images to be processed
        :return: A DataFrame containing the available data
        """
        data_frame = None
        responses = []
        # Check if specified folder exists
        if os.path.isdir(folder_name):
            # Get the images if the exist and if they are in the supported types
            images = [filename for filename in os.listdir(folder_name)
                      if os.path.isfile(os.path.join(folder_name, filename)) and
                      filename.split('.')[-1].lower() in self.IMAGE_FILE_TYPES]

            payload = {}
            payload['requests'] = []
            # Create a list to associate responses with images
            images_names = list()
            for iterator, image_file in enumerate(images):
                image_path = os.path.join(folder_name, image_file)
                images_names.append(self.path_leaf(image_path))
                self.images_names.append(self.path_leaf(image_path))
                with open(image_path, 'rb') as image:
                    image_content = base64.b64encode(image.read())
                    image_payload = {}
                    image_payload['image'] = {}
                    image_payload['image']['content'] = image_content.decode('UTF-8')
                    image_payload['features'] = [{
                                'type': 'LABEL_DETECTION',
                                'maxResults': 5
                    }]
                    payload['requests'].append(image_payload)

                    # Need to check if we have reached 10 images, meaning we must send a request,
                    # Google does not like more than 10 images (more or less) per request
                    if (iterator + 1) % 10 == 0:
                        service_request = self.google_vision_service.images().annotate(body=payload)
                        payload = {}
                        payload['requests'] = []
                        try:
                            response = service_request.execute()
                            intermediate = response['responses']
                            merged = zip(images_names, intermediate)
                            response['responses'] = []
                            for element in merged:
                                image_labeled = {}
                                image_labeled[element[0]] = element[1]
                                response['responses'].append(image_labeled)
                            responses.append(response)
                            images_names = list()
                            # Add one second delay before making another request
                            time.sleep(5)
                        except (HttpError, HttpLib2Error) as ex:
                            print('The following error occurred trying to label images with Google {0}'.format(str(ex)))
                            continue

            # This is just in case images were less than 10 or the remaining of more than any multiple of 10
            service_request = self.google_vision_service.images().annotate(body=payload)
            try:
                response = service_request.execute()
                intermediate = response['responses']
                merged = zip(images_names, intermediate)
                response['responses'] = []
                for element in merged:
                    image_labeled = dict()
                    image_labeled[element[0]] = element[1]
                    response['responses'].append(image_labeled)
                responses.append(response)
            except (HttpError, HttpLib2Error) as ex:
                print('The following error occurred trying to label images with Google {0}'.format(str(ex)))

            finally:
                # Iterate the responses and construct one single dictionary with one response key only
                final_response = dict()
                final_response['responses'] = []
                for partial in responses:
                    labels = partial['responses']
                    final_response['responses'].append(labels)
                google_results = dict()
                # Process the dictionary and get the DataFrame with results
                for tagged_responses in final_response['responses']:
                    tags_found = []
                    # Get the labels
                    for image_tagged in tagged_responses:
                        # Get the labels for this image
                        labels = image_tagged.values()
                        for label in labels[0]['labelAnnotations']:
                            tag_found = (label['description'], label['score'])
                            tags_found.append(tag_found)
                        google_results[image_tagged.keys()[0]] = tags_found

                data_series = pandas.Series(google_results, index=self.images_names, name='GoogleVision')
                data_frame = pandas.DataFrame(data_series, index=self.images_names, columns=self.apis)
                return data_frame

        else:
            raise ValueError('The input directory does not exist: %s' % folder_name)

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



