__author__ = 'internetmosquito'
# test_meteogalicia_helper.py
import datetime
import unittest
from image_tagging import ImageTagger


class ImageTaggerTests(unittest.TestCase):

    ############################
    #### setup and teardown ####
    ############################
    helper = None

    # executed prior to each test
    def setUp(self):
        self.tagger = self.create_tagger()

    # executed after to each test
    def tearDown(self):
        self.helper = None

    ########################
    #### helper methods ####
    ########################
    def create_tagger(self):
        tagger = ImageTagger()
        return tagger

    def configure_tagger(self, tagger):
        #Create the required params
        config_file = 'config.yml'
        tagger.configure_tagger(config_file)


    ###############
    #### tests ####
    ###############
    def test_not_configured_tagger_cannot_query(self):
        print 'Checking helper is not configured...'
        self.assertFalse(self.helper.configured)
        print 'Checking that a non-configured helper must return None upon calling query'
        self.assertIsNone(self.helper.query())

    def test_configured_tagger_can_query(self):
        self.configure_helper(self.helper)
        print 'Checking helper is configured...'
        self.assertTrue(self.helper.configured)
        print 'Checking that a configured helper does query and returns something...'
        json_response = self.helper.query()
        self.assertIsNotNone(json_response)

    def test_mis_configured_helper_returns_error(self):
        self.misconfigure_helper(self.helper)
        print 'Checking helper is misconfigured...'
        self.assertTrue(self.helper.configured)
        print 'Checking that a misconfigured helper returns error on query'
        json_response = self.helper.query()
        self.assertIn('exception', json_response)

    def test_configured_helper_returns_forecast_data(self):
        self.configure_helper(self.helper)
        self.assertTrue(self.helper.configured)
        forecast_data = self.helper.get_forecast_data()

        #Try to load the object to see if it is valid
        try:
            if forecast_data:
                #Convert the list of objects to a JSON string
                forecast_json = self.helper.get_json_from_points(forecast_data)
                print forecast_json
                #Check that the json dict contains the keys -> Point -> Days -> Variables -> Values and has data
                if forecast_json['points'][0]['days'][1]['variables']:
                    values_list = forecast_json['points'][0]['days'][1]['variables'][0]
                    print 'Checking that a forecast obtained data at least has some values'
                    self.assertTrue(values_list)
                else:
                    print 'No points available'
                    return False
            else:
                print 'MeteoGalicia returned an empty list'
                return False
        except ValueError:
            return False

    def test_forecast_data_converts_to_json_string(self):
        self.configure_helper(self.helper)
        self.assertTrue(self.helper.configured)
        forecast_data = self.helper.get_forecast_data()
        #Try to load the object to see if it is valid
        try:
            if forecast_data:
                #Convert the list of objects to a JSON string
                forecast_json = self.helper.get_json_from_points(forecast_data)
                #Convert the dict to a json_string
                json_str = self.helper.get_string_from_json(forecast_json)
                print 'Checking that JSON string obtained is String and not empty'
                self.assertIsInstance(json_str, str)
                self.assertTrue(json_str)
            else:
                print 'MeteoGalicia returned an empty list'
                return False
        except Exception, e:
            print 'MeteoGalicia returned an empty list'
            return False

    def test_configured_helper_returns_forecast_data_since_midnight(self):
        self.configure_helper(self.helper)
        self.assertTrue(self.helper.configured)
        forecast_data = self.helper.get_forecast_data()

        try:
            if forecast_data:
                forecast_json = self.helper.get_json_from_points(forecast_data)
                #Check that the json dict contains the keys -> Point -> Days -> Variables -> Values and has data
                if forecast_json['points'][0]['days'][0]:
                    # starts time in ISO format '2015-11-26 00:00:00+01:00'
                    midnight_time = datetime.datetime.combine(datetime.date.today(), datetime.time(0, 0)).isoformat(' ')
                    print 'Checking that results for today (the first day) start from 00:00:00'
                    self.assertIn(midnight_time, forecast_json['points'][0]['days'][0]['beginning'])
                else:
                    print 'No points available'
                    return False
            else:
                print 'MeteoGalicia returned an empty list'
                return False
        except ValueError:
            return False

if __name__ == "__main__":
    unittest.main()
