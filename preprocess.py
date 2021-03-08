import os
import argparse
from pycorenlp import StanfordCoreNLP
import unidecode


class Preprocess:
    def __init__(self, input_path, output_path, annotators):
        """
        Start Stanford CoreNLP Server
        Initialize input, output_folder, standford_annotators, log_path (optional)
        "input" can be a file or a folder
        """
        # Start Stanford CoreNLP Server
        self.nlp = StanfordCoreNLP('http://localhost:9000')

        # Initialize input, output_folder, standford_annotators, log_path (optional)
        self.input = os.path.abspath(input_path)
        self.output_folder = os.path.abspath(output_path)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.log_path = os.path.abspath('')
        self.standford_annotators = annotators
        if os.path.isdir(self.input):
            self.input_type = "dir"
        elif os.path.isfile(self.input):
            self.input_type = "file"

        #print("Your Input: " + self.input + ", " + self.input_type)
        #print("Your Output Folder: " + self.output_folder)
        #print("Your Stanford annotators: " + self.standford_annotators)

    def str_process(self, row_string):
        """
        This Standford CoreNLP package requires the text input as 1 single string.
        The input annotators are in you command line input.
        :param row_string: The string format input for Standford CoreNLP
        :return: Json format output
        """
        processed_json = self.nlp.annotate(row_string, properties={
            'annotators': self.standford_annotators,
            'outputFormat': 'json'
        })
        return processed_json

    def output_preprocessed_data(self, json_input, file_name):
        """
        Output preprocessed data into a file.
        :param json_input: json formatted data generated from function str_process
        :param file_name: output file name
        :return: None
        """
        rows = []
        for sent in json_input['sentences']:
            parsed_sent = " ".join([t['originalText'] + "/" + t['pos'] for t in sent['tokens']])
            rows.append(parsed_sent)
        output_file_path = self.output_folder + '/' + file_name
        if os.path.exists(output_file_path):
            open(output_file_path, 'w').close()
        with open(output_file_path, 'a') as preprocessed_out:
            for r in rows:
                preprocessed_out.write(unidecode.unidecode(r) + "\n")

    def pos_tagging(self):
        """
        Read an input file/folder as raw data input.
        Output the results into files witH POS Tags.
        :return: None
        """
        if self.input_type == "file":
            file_name = os.path.basename(self.input)
            text_string = ""
            with open(self.input, 'rb') as file_input:
                for r in file_input:
                    text_string = " ".join([text_string, r.strip().decode('utf-8', 'backslashreplace')])
            #print('\t\t' + self.input + " Done!")
            parsed_json = self.str_process(text_string)
            self.output_preprocessed_data(parsed_json, file_name)
        elif self.input_type == "dir":
            for file_name in os.listdir(self.input):
                input_file_path = self.input + "/" + file_name
                text_string = ""
                with open(input_file_path, 'rb') as file_input:
                    for r in file_input:
                        text_string = " ".join([text_string, r.strip().decode('utf-8', 'backslashreplace')])
                parsed_json = self.str_process(text_string)
                #print('\t\t' +input_file_path + " Done!")
                self.output_preprocessed_data(parsed_json, file_name)
