import os
import yaml
import json
import requests
from requests.auth import HTTPBasicAuth
from simplejson import JSONDecodeError


class ImaggaHelper(object):
    """
    A class that provides commodity methods to access imagga, mostly based on their tutorial that can
    be found here:
    http://imagga.com/blog/batch-image-processing-from-local-folder-using-imagga-api/
    """
    IMAGGA_API_KEY = ''
    IMAGGA_API_SECRET = ''
    ENDPOINT = 'https://api.imagga.com/v1'
    IMAGGA_FILE_TYPES = ['png', 'jpg', 'jpeg', 'gif']

    def __init__(self):
        self.auth = None
        self.configured = False

    def configure_imagga_helper(self, config_file):
        """
        Reads the API credentials from the specified YAML file and initializes API clients
        :param config_file: The file path to the config YAML file
        :return: True if config file was parsed and API clients initialized correctly
        """
        # Check if provided config yaml file actually does exist
        if os.path.isfile(config_file):
            config = yaml.safe_load(open(config_file))
            # Get config data
            self.IMAGGA_API_KEY = config['imagga']['api-key']
            self.IMAGGA_API_SECRET = config['imagga']['api-secret']
            if self.IMAGGA_API_KEY and self.IMAGGA_API_KEY:
                self.auth = HTTPBasicAuth(self.IMAGGA_API_KEY, self.IMAGGA_API_SECRET)
                self.configured = True
                return True
        else:
            return False

    def upload_image(self, image_path):
        """
        Uploads an image to the Imagga API so it can be processed afterwards
        :param image_path: The full path of the image to be uplodad
        :return: The content ID associated with the uploaded image
        """
        if not os.path.isfile(image_path):
            raise ValueError('Provided image path is invalid, cannot upload to Imagga')

        # Open the desired file
        with open(image_path, 'rb') as image_file:
            filename = image_file.name

            # Upload the multipart-encoded image with a POST
            # request to the /content endpoint
            content_response = requests.post(
                '%s/content' % self.ENDPOINT,
                auth=self.auth,
                files={filename: image_file})

            # Example /content response:
            # {'status': 'success',
            #  'uploaded': [{'id': '8aa6e7f083c628407895eb55320ac5ad',
            #                'filename': 'example_image.jpg'}]}
            try:
                uploaded_files = content_response.json()['uploaded']
            except JSONDecodeError:
                import pdb
                pdb.set_trace()
                return None

            # Get the content id of the uploaded file
            content_id = uploaded_files[0]['id']

        return content_id

    def tag_image(self, image, verbose=False):
        """
        Calls the Imagga tagging API endpoint and returns the results
        :param image: Can be a URL or the content ID from uploading the image
        :param verbose: If true it includes the origin of the tagging procedure
        :return: The JSON response from the tagging call
        """
        # Using the content id and the content parameter,
        # make a GET request to the /tagging endpoint to get
        # image tags
        tagging_query = {
            'content': image,
            'verbose': verbose,
        }
        tagging_response = requests.get(
            '%s/tagging' % self.ENDPOINT,
            auth=self.auth,
            params=tagging_query)

        # In case we want to save to file the results with a decent format
        # with open('results.json', 'w') as out:
        #    res = json.dump(tagging_response.json(),
        #                    out,
        #                    sort_keys=True,
        #                    indent=4,
        #                    separators=(',', ': '))

        return tagging_response.json()

    def tag_folder(self, folder_path):
        """
        Iterates over the images found in the specified path and calls Imagga API for each image
        :param folder_path: The full path of the folder to extract and process images from
        :param verbose: If true it includes the origin of the tagging procedure
        :return: The JSON response from the tagging call
        """
        import pdb
        pdb.set_trace()
        results = {}
        # Check if specified folder exists
        if os.path.isdir(folder_path):
            # Get the images if the exist and if they are in the supported types
            images = [filename for filename in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, filename)) and
                      filename.split('.')[-1].lower() in self.IMAGGA_FILE_TYPES]

            images_count = len(images)
            for iterator, image_file in enumerate(images):
                image_path = os.path.join(folder_path, image_file)
                print('[%s / %s] %s uploading' %
                      (iterator + 1, images_count, image_path))

                content_id = self.upload_image(image_path)
                if content_id:
                    tag_result = self.tag_image(content_id, True)
                    results[image_file] = tag_result
                    print('[%s / %s] %s tagged' %
                          (iterator + 1, images_count, image_path))
        else:
            raise ValueError('The input directory does not exist: %s' % folder_path)
        import pdb
        pdb.set_trace()
        response = json.dumps(results, ensure_ascii=False, indent=4).encode('utf-8')
        return response
