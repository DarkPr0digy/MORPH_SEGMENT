import os
import sys

import Levenshtein as LevenshteinDistance

from morphology.Symbols import Symbols


class DataCleaner:
    """
    Class that does the data cleaning
    """

    def __init__(self, filename: str):
        """
        Constructor Method
        :param filename: name of input file
        """
        # Open file for reading only
        self.file = open(os.path.join(sys.path[0], filename), "r")
        # self.file = open(filename, "r")
        self.lines = self.file.readlines()
        self.s = Symbols()

    def reformat(self, filename: str):
        """
        Method Used to Clean the Data contained in the original files, and to generate the surface segmentation
        :param filename: output file
        """
        new_file = open(os.path.join(sys.path[0], filename + ".conll"), "w")
        # open new file for writing

        for line in self.lines:
            is_blank = False
            if line == '' or line == '\n' or line == ' ':
                is_blank = True
            # Check if line is whitespace
            line_value = line.split("\t")

            try:
                is_int = isinstance(int(line_value[0]), int)
            except ValueError:
                is_int = False
            # check if line is int
            try:
                is_float = isinstance(float(line_value[0]), float)
            except ValueError:
                is_float = False
            # Check if line is float

            if not self.s.inArr(line_value[0]) and not is_int and not is_float and not is_blank \
                    and not len(line_value[0]) == 1:
                # Formats as follows:
                # word | surface segmented form | orthographic segmented form
                orthographic_form = normaliseOrthographicForm(line_value[3].rstrip('\n'))
                surface_segmented = generateSurfaceSegmentation(removeLabels(line_value[0]),
                                                                removeLabels(line_value[3]))
                #########################################
                if label_per_morpheme(orthographic_form) and not \
                        has_insert(LevenshteinDistance.editops(removeLabels(orthographic_form), surface_segmented)):
                    labelled_surface_seg = generateLabelledSurfaceSegmentation(surface_segmented, orthographic_form)
                    new_file.write(removeLabels(line_value[0]) + " | " +
                                   surface_segmented + " | " + labelled_surface_seg
                                   + " | " + orthographic_form + '\n')
                else:
                    continue

        # Close both files to avoid leakages
        self.file.close()
        new_file.close()


def normaliseOrthographicForm(orthographic: str):
    """Method to normalise the format of the orthographic form to make using it as input easier for the CRF"""
    # Formats orthographic form into following format
    # aa[bb]cc[dd]
    str2_arr = []

    # Removes Labels at the front of orthographic forms
    if orthographic[0] == '[':
        beginningLabel = True
        while beginningLabel:
            index = orthographic.find(']')
            orthographic = orthographic[index + 1: len(orthographic)]
            if orthographic[0] == '[':
                continue
            else:
                beginningLabel = False

    # Removes extra character
    for char in orthographic:
        if char == '$' or char == '-':
            continue
        else:
            str2_arr.append(char)

    # Combines double labels
    label = []
    str = []
    tag = False
    for i in range(len(str2_arr)):
        try:
            double_label = str2_arr[i] == ']' and str2_arr[i + 1] == '['
        except IndexError:
            double_label = False

        if str2_arr[i] == '[':
            tag = True
        elif double_label:
            label.append('|')
        elif str2_arr[i] == ']':
            tag = False
            str.append('[')
            for char in label:
                str.append(char)
            str.append(']')
            label = []
        elif tag:
            label.append(str2_arr[i])
        else:
            str.append(str2_arr[i])

    tmp_ortho = "".join(str)

    is_label = False
    is_first_label_char = False
    str = []

    for char in tmp_ortho:
        if char == "[":
            is_label = True
            is_first_label_char = True
            str.append(char)
        elif is_label and is_first_label_char:
            str.append(char.upper())
            is_first_label_char = False
        elif is_label and char == "|":
            is_first_label_char = True
            str.append(char)
        elif char == "]":
            is_label = False
            str.append(char)
        elif is_label and char == "+":
            str.append("|")
        else:
            str.append(char)

    return "".join(str)


def removeLabels(str2: str):
    """Method to remove labels from the orthographic segmentation so this form
    can be used to generate the surface segmentation"""
    str2_arr = []
    last_seen_bracket = []
    for char in str2:
        if char == "[":
            last_seen_bracket.append(char)
            str2_arr.append("-")
        elif char == "(":
            last_seen_bracket.append(char)
        elif char == ")" or char == "]":
            if len(last_seen_bracket) >= 1:
                last_seen_bracket.pop()
            else:
                continue
        elif char == "-" or char == '$':
            continue
        elif len(last_seen_bracket) >= 1:
            continue
        else:
            str2_arr.append(char)

    if len(str2_arr) > 1:
        for i in range(len(str2_arr)):
            try:
                if str2_arr[i] == "-" and str2_arr[i - 1] == "-":
                    str2_arr.pop(i - 1)
                    # Some segments have dual purpose, so this removes dual dashes that result from this
            except IndexError:
                continue

        if str2_arr[len(str2_arr) - 1] == "\n":
            str2_arr.pop()

    return "".join(str2_arr).rstrip("-").lstrip("-")


def de_segment(word: str):
    """Method used to de-segment the orthographic form of a word"""
    ans = ""
    for char in word:
        if char != "-":
            ans += char
    return ans


def label_per_morpheme(orthographic: str):
    if orthographic.find('[') == -1:
        return False
    labels = []
    str = []
    orthographic += "|"
    is_label = False
    is_str = True
    tmp_label = ''
    tmp_str = ''
    for char in orthographic:
        if char == '[':
            if tmp_str:
                str.append(tmp_str)
            tmp_str = ''
            is_label = True
            is_str = False
        elif char == ']':
            if tmp_label:
                labels.append(tmp_label)
            tmp_label = ''
            is_label = False
            is_str = True
        elif char == "|":
            if is_str:
                if tmp_str:
                    str.append(tmp_str)
            elif is_label:
                if tmp_label:
                    labels.append(tmp_label)
        elif is_str:
            tmp_str += char
        elif is_label:
            tmp_label += char

    return len(labels) == len(str)


def generateLabelledSurfaceSegmentation(surface_segmented: str, orthographic_labelled: str):
    """
    Method to generate labelled surface segmentation for a word
    :param surface_segmented: surface segmented form of word
    :param orthographic_labelled: orthographic labelled form of word
    :return: surface labelled form of word
    """
    if removeLabels(orthographic_labelled) == surface_segmented:
        return orthographic_labelled
    else:
        surface = surface_segmented.split('-')
        ortho = removeLabels(orthographic_labelled).split('-')

        if len(surface) == len(ortho):
            # Segments are the same, labels can be directly translated
            labels = get_labels(orthographic_labelled)
            string = []

            for i in range(len(surface)):
                string.append(surface[i] + '[' + labels[i] + ']')
            return "".join(string)
        else:
            labels = get_labels(orthographic_labelled)

            editops = LevenshteinDistance.editops(removeLabels(orthographic_labelled), surface_segmented)
            source = [pos for pos in removeLabels(orthographic_labelled)]
            outputword = [pos for pos in removeLabels(orthographic_labelled)]
            destination = [pos for pos in surface_segmented]
            dash_pos = [x for x in range(len(outputword)) if outputword[x] == '-']

            edit_counter = 0
            for edit in editops:
                labelPos = 0
                for x in dash_pos:
                    if edit[1] >= x:
                        labelPos += 1
                deleted_dash_not_first_label, double_dash, deleted_dash_first_label = False, False, False
                if edit[0] == 'delete':
                    if outputword[edit[1]] == '-':
                        if outputword[edit[1] + 1] == '-':
                            double_dash = True
                        elif not all_stars(outputword[0: edit[1]]):
                            deleted_dash_not_first_label = True
                        elif all_stars(outputword[0: edit[1]]):
                            deleted_dash_first_label = True
                    outputword[edit[1]] = '*'

                if edit[0] == 'replace':
                    outputword[edit[1]] = destination[edit[2]]
                if edit[0] == 'insert':
                    outputword.insert(edit[2], destination[edit[1]])

                if deleted_dash_not_first_label:
                    if outputword[edit[1] + 1] in alphabet and previous_segment(outputword[:edit[1]]):
                        # tmp = labels[labelPos]
                        del labels[labelPos]
                        # labels[labelPos - 1] += '|' + tmp

                segmentedArray = printSegments(outputword)
                for i, segment in enumerate(segmentedArray):
                    if segment == '' and i == 0:
                        tmp = labels[0]
                        del labels[0]
                        labels[0] = tmp + "|" + labels[0]
                    elif segment == '':
                        # label = labels[i]
                        # labels[i - 1] = labels[i - 1] + '|' + label
                        del labels[i]
                edit_counter += 1
                dash_pos = [x for x in range(len(outputword)) if outputword[x] == '-']

            string = []
            for x in range(len(surface)):
                try:
                    string.append(surface[x] + '[' + labels[x] + ']')
                except IndexError:
                    print(editops)
                    print(orthographic_labelled)
                    print(''.join(source))
                    print(surface_segmented)
                    print(labels)
                    exit(0)
            return "".join(string)


def generateSurfaceSegmentation(word: str, orthographic_form: str):
    """
    Method used to generate the surface segmentation of a word
    given the word and the orthographic form of the word
    :param word: the word itself
    :param orthographic_form: the orthographic / canonical form of the word
    :return: surface segmented form of word
    """

    if word.lower() == de_segment(orthographic_form).lower():
        # If the word and the orthographic form are the same,
        # then the orthographic form is the surface form and this can be returned
        return orthographic_form
    else:
        replace = []
        # Generate list of operations needed to turn de-segmented orthographic form into original word
        # of the form [(operation, source pos, destination pos)...]
        edits = LevenshteinDistance.editops(de_segment(orthographic_form), word)

        for x in edits:
            if x[0] == 'replace':
                replace.append(True)
            else:
                replace.append(False)
        if all(replace):
            # If all operations being performed on a word are replace operations then one can simply add dashes to
            # the word where they appear in the orthographic form to generate the surface form

            # Get position of all dashes in the orthographic form
            dash_pos = [pos for pos in range(len(orthographic_form)) if orthographic_form[pos] == '-']
            arr = list(word)
            for d in dash_pos:
                arr.insert(d, "-")
            return "".join(arr)

        else:
            segmented_form = list(orthographic_form)
            # Get position of all dashes in the segmented form
            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
            de_segmented_form = list(de_segment(orthographic_form))

            for ops in edits:
                # Iterate through operations one by one and try to reverse them
                if ops[0] == 'delete':
                    # Get position of the change
                    position = ops[2]

                    # Remove from position in de-segmented form
                    del de_segmented_form[position]

                    # Position of same char in segmented form will be higher if there are dashes that occur before it
                    # This loop accounts for that difference
                    for x in dash_pos:
                        if position >= x:
                            position += 1

                    try:
                        # try determine if the segment is an lone character such as
                        # ...-x-...
                        lone = segmented_form[position - 1] == '-' and segmented_form[position + 1] == '-'
                    except IndexError:
                        continue

                    # Remove from position in segmented form
                    del segmented_form[position]

                    if lone:
                        # if it is a lone character, also delete the preceding dash
                        del segmented_form[position]

                    # Update positions of the dashes
                    dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
                elif ops[0] == 'replace':
                    # Nothing needs to be done to de-segmented form because
                    # this has no net effect on construction of the word

                    # Get position of the change
                    position = ops[2]

                    # Position of same char in segmented form will be higher if there are dashes that occur before it
                    # This loop accounts for that difference
                    for x in dash_pos:
                        if position >= x:
                            position += 1
                    try:
                        if segmented_form[position - 1] == '-' and segmented_form[position + 1] == '-':
                            # if the changed character is a lone character delete preceding dash
                            del segmented_form[position - 1]

                            # Update position of dashes
                            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
                    except IndexError:
                        continue

            # Final update of position of dashes
            dash_pos = [pos for pos in range(len(segmented_form)) if segmented_form[pos] == '-']
            surface_segmented = list(word)
            for dash in dash_pos:
                # where there is a dash in updated segmented form put a dash in the word
                surface_segmented.insert(dash, '-')

            return "".join(surface_segmented).rstrip("-").lstrip("-")


def get_labels(orthographic: str):
    """
    Method to get the labels from the orthographic form of the word
    :param orthographic: the orthographic form of the word with labels included
    :return: a list of the labels in the word
    """
    labels = []
    tmp = ''
    tag = False

    # Get all labels from orthographic form
    for char in orthographic:
        if char == '[':
            tag = True
        elif char == ']':
            labels.append(tmp)
            tag = False
            tmp = ''
        elif tag:
            tmp += char
    return labels


alphabet = 'abcdefghijklmnopqrstuvwxyz'


def arrToWord(word):
    """
    Helper method for surface labelled method
    :param word: the segmented word with *s
    :return: a cleaned version of the word
    """
    finalword = ''
    for w in word:
        if w == '*':
            pass
        else:
            finalword += w
    return finalword


def printSegments(word):
    """
    Helper method for surface labelled method
    :param word: the word with *s removed
    :return: the word
    """
    outword = arrToWord(word)
    finalword = outword.split('-')
    return finalword


def all_stars(arr: [str]):
    """
    Helper method to check if all characters in an array are *s
    :param arr: arrray of strings
    :return: boolean
    """
    for char in arr:
        if char == '*':
            continue
        else:
            return False
    return True


def previous_segment(arr: [str]):
    """
    Helper method to check if previous segment is valid
    :param arr: arr of strings
    :return: boolean
    """
    for i in range(len(arr)):
        char = arr[len(arr) - 1 - i]
        if char == '*':
            continue
        if char in alphabet:
            return True
        if char == '-':
            return False
    return False


def has_insert(edit_ops: [(str, int, int)]):
    """
    Helper method to check if one of the edit operations is edit
    :param edit_ops: list edit tuples
    :return: boolean
    """
    for edit in edit_ops:
        if edit[0] == 'insert':
            return True
    return False

    def get_ortho_labels(orthographic: str):
        labels = []
        tmp = ''
        tag = False

        # Get all labels from orthographic form
        for char in orthographic:
            if char == '[':
                tag = True
            elif char == ']':
                labels.append(tmp)
                tag = False
                tmp = ''
            elif tag:
                tmp += char
        return labels


def get_surface_segments(ortho: str):
    """
    Method to extract the segments from the orthographic form of the word
    :param orthographic: the orthographic form of the word
    :return: list of all the segments in the word
    """
    segments = []
    tmp = ''
    label = False

    # Get all segments from orthographic form
    for char in ortho:
        if char == '[':
            segments.append(tmp)
            tmp = ''
            label = True
        elif char == ']':
            label = False
        elif not label:
            tmp += char
    return segments


def deletion_merge(edit_ops: [(str, int, int)], position: int):
    """
    Check if deletion(s) followed by reokacement for merge
    :param edit_ops: list of edit operation tuples
    :param position: position in list
    :return: boolean
    """
    for i in range(len(edit_ops[position:])):
        # check deletions and replacement
        if edit_ops[i][0] == 'delete':
            edit_pos = i
            deletion_counter = 1
            while edit_ops[edit_pos][0] == 'delete':
                if edit_ops[edit_pos + 1][0] == 'replace':
                    if edit_ops[position][1] + deletion_counter == edit_ops[edit_pos + 1][1]:
                        return True
                deletion_counter += 1
                edit_pos += 1
    return False


languages = ["zulu", "swati", "ndebele", "xhosa"]

for lang in languages:
    print("Language: " + lang)
    inputFile = DataCleaner(lang + '/' + lang + ".unique.train.conll")
    inputFile.reformat(lang + '/' + lang + ".clean.train")
    inputFile = DataCleaner(lang + '/' + lang + ".test.conll")
    inputFile.reformat(lang + '/' + lang + ".clean.test")
    print(lang + " cleaning complete.\n#############################################")
