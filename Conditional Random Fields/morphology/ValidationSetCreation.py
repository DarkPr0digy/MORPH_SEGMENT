class ValidationSet:
    """
    Class that creates the validation set
    """

    def __init__(self, filename: str):
        """
        Constructor for class
        :param filename: name of file to create dev set from
        """
        self.file_name = filename
        self.training_file = open(filename, "r")
        self.lines = self.training_file.readlines()
        self.line_count = len(self.lines)

    def create_validation_set(self, filename: str):
        """
        Method that creates validation set
        :param filename: name of dev set file
        """
        """Method used to develop the dev set from the training set"""
        validation_file = open(filename, "w")
        training_file_updated_content = ""

        # Take 10% of training set to make validation set
        limit = self.line_count * 0.1
        limit = limit.__round__()
        print(limit)

        num_entries = 0
        counter = 0
        for line in self.lines:
            # Could change this to make it non-deterministic
            if counter % 10 == 0:
                validation_file.write(line)
                num_entries += 1
            else:
                training_file_updated_content += line
            if num_entries > limit:
                break
            counter += 1

        self.training_file.close()

        # Write updated content to training file
        self.training_file = open(self.file_name, "w")
        self.training_file.write(training_file_updated_content)
        self.training_file.close()
        validation_file.close()


languages = ["zulu", "swati", "ndebele", "xhosa"]

for lang in languages:
    print("Language: " + lang)
    file_name = lang + '/' + lang + ".clean.train.conll"
    # print(file_name)
    inputFile = ValidationSet(file_name)
    file_name = lang + '/' + lang + ".clean.dev.conll"
    # print(file_name)
    inputFile.create_validation_set(file_name)
    print(lang + " validation set complete.\n#############################################")
