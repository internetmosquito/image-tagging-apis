import json
import unittest
import mock
from mock import Mock, patch
import yaml
import os

from image_tagging import ImageTagger
from imagga import ImaggaHelper


def mocked_requests_post(*args, **kwargs):
        """
        Used to mock the post response from uploading images to Imagga
        :param args:
        :param kwargs:
        :return:
        """
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code

            def json(self):
                return self.json_data

        if args[0] == 'https://api.imagga.com/v1/content':
            # Get the filename tried to uploaded
            response = {"status": "success",
                        "uploaded": [{"id": "4598e39043b2f7bbef85a44422fbd824",
                                       "filename": "sea-man-person-surfer.jpg"}]}
            return MockResponse(response, 200)

        return MockResponse({}, 404)


class ImageTaggerTests(unittest.TestCase):

    ############################
    #### setup and teardown ####
    ############################
    helper = None

    # executed prior to each test
    def setUp(self):
        self.tagger = self.create_tagger()
        self.imagga_helper = self.create_imagga_helper()

    # executed after each test
    def tearDown(self):
        self.helper = None

    ########################
    #### helper methods ####
    ########################
    def create_tagger(self):
        tagger = ImageTagger()
        return tagger

    def create_imagga_helper(self):
        imagga = ImaggaHelper()
        return imagga

    def configure_tagger(self, config_file, tagger):
        #Create the required params
        tagger.configure_tagger(config_file)

    def configure_imagga_helper(self, config_file, imagga_helper):
        #Create the required params
        imagga_helper.configure_imagga_helper(config_file)



    ###############
    #### tests ####
    ###############
    def test_malformed_yaml_misconfigures_tagger(self):
        print 'Checking tagger is not configured...'
        # Empty config data

        d = {
            'visual-recognition': {
                'api-key': ''
            },
            'clarifai': {
                'client-id': '',
                'client-secret': ''
            }
        }
        with open('dummy_config.yml', 'w') as yaml_file:
            yaml_file.write(yaml.dump(d, default_flow_style=False))
        self.configure_tagger('dummy_config.yml', self.tagger)
        self.assertFalse(self.tagger.configured)
        os.remove('dummy_config.yml')

    def test_valid_yaml_configures_tagger(self):
        print 'Checking tagger is well configured...'
        self.configure_tagger('config.yml', self.tagger)
        self.assertTrue(self.tagger.configured)

    def test_configured_tagger_returns_data_from_visual_recognition_api(self):
        print 'Checking tagger queries visual recognition API'
        # Configure the mock to return a response with some dummy json response
        self.configure_tagger(config_file='config.yml', tagger=self.tagger)
        with patch('image_tagging.VisualRecognitionV3.classify') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_vr_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            response = self.tagger.process_images_visual_recognition('whatever', store_results=False)
            self.assertIsNotNone(response)
            self.assertTrue(response.empty)

    def test_configured_tagger_returns_data_from_clarifai(self):
        print 'Checking tagger queries clarifai API'
        # Configure the mock to return a response with some dummy json response
        self.configure_tagger(config_file='config.yml', tagger=self.tagger)
        # Patch the actual code that clarifai client uses
        with patch('clarifai.client.mime_util.post_multipart_request') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_clarifai_result.json', 'r')
            # Read as string
            dummy_response = fd.read()
            fd.close()
            mock_get.return_value = dummy_response
            response = self.tagger.process_images_clarifai(folder_name='sample_images')
            self.assertIsNotNone(response)
            # Check returned DataFrame has 16 rows, as expected
            self.assertEqual(16, len(response.index))

    def test_configured_tagger_returns_data_from_imagga(self):
        pass

    def test_configured_imagga_wrapper_has_credentials(self):
        print 'Checking imagga helper has credentials upon configuration'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        self.assertIsNotNone(self.imagga_helper.IMAGGA_API_KEY)
        self.assertIsNotNone(self.imagga_helper.IMAGGA_API_SECRET)

    @mock.patch('imagga.requests.post', side_effect=mocked_requests_post)
    def test_configured_imagga_wrapper_can_upload_image(self, mock_post):
        print 'Checking imagga helper can upload imaga to Imagga'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        # Patch the upload_image method so we don't really call it
        #with patch('imagga.requests.post', side_effect=mocked_requests_post) as mock_get:
            # Configure the mock to return a response with an OK status code.

            # dummy_response = '''{"status": "success", "uploaded":
            #                     [{"id": "4598e39043b2f7bbef85a44422fbd824",
            #                     "filename": "sea-man-person-surfer.jpg"}]}'''
        # mock_get.return_value = dummy_response
        response = self.imagga_helper.upload_image(image_path='sample_images/sea-man-person-surfer.jpg')
        self.assertIsNotNone(response)
        self.assertEqual(response, u'4598e39043b2f7bbef85a44422fbd824')

    def test_configured_imagga_wrapper_can_tag_image(self):
        print 'Checking imagga helper can tag Image'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        # Patch the upload_image method so we don't really call it
        with patch('imagga.ImaggaHelper.tag_image') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_imagga_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            response = self.imagga_helper.tag_image('whatever')
            self.assertIsNotNone(response)
            self.assertEqual(response['results'][0]['image'], '03204bdaaa3301fb8ce8a4c1e7a1ff17')

if __name__ == "__main__":
    unittest.main()
