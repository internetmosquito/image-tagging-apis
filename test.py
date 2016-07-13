import json
import unittest
from mock import Mock, patch
import yaml
import os

from image_tagging import ImageTagger
from imagga import ImaggaHelper


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
        with patch('image_tagging.ClarifaiApi.tag_images') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_clarifai_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            response = self.tagger.process_images_clarifai('whatever')
            self.assertIsNotNone(response)
            self.assertTrue(response.empty)

    def test_configured_tagger_returns_data_from_imagga(self):
        pass

    def test_configured_imagga_wrapper_has_credentials(self):
        print 'Checking imagga helper has credentials upon configuration'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        self.assertIsNotNone(self.imagga_helper.IMAGGA_API_KEY)
        self.assertIsNotNone(self.imagga_helper.IMAGGA_API_SECRET)

    def test_configured_imagga_wrapper_can_upload_image(self):
        print 'Checking imagga helper can upload imaga to Imagga'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        # Patch the upload_image method so we don't really call it
        with patch('imagga.ImaggaHelper.upload_image') as mock_get:
            # Configure the mock to return a response with an OK status code.

            dummy_response = '''{"status": "success", "uploaded":
                                [{"id": "4598e39043b2f7bbef85a44422fbd824",
                                "filename": "sea-man-person-surfer.jpg"}]}'''
            import pdb
            pdb.set_trace()
            mock_get.return_value = dummy_response
            response = self.imagga_helper.upload_image(image_path='whatever')
            self.assertIsNotNone(response)
            content_id = json.loads(response)['uploaded'][0]['id']
            self.assertEqual(content_id, u'4598e39043b2f7bbef85a44422fbd824')

        # self.imagga_helper.upload_image(image_path='sample_images/sea-man-person-surfer.jpg')


if __name__ == "__main__":
    unittest.main()
