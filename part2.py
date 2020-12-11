import sys
import time
from pathlib import Path


def addCount(parent, child, d):
    # Increment the count of [parent][child] in dictionary d
    if parent in d:
        if child in d[parent]:
            d[parent][child] += 1
        else:
            d[parent][child] = 1
    else:
        d[parent] = {child: 1}


def getEmissions(file, k=0.5):
    """
    input = training file
    output = emission parameters (dict)
    @param k: Words appearing less than k times will be
    replaced with #UNK#
    dict format = {i: {o:emission prob}}
    """
    emissions = {}
    count = {}

    with open(file, encoding="utf-8") as f:
        for line in f:
            temp = line.strip()

            # ignore empty lines
            if len(temp) == 0:
                continue
            else:
                last_space_index = temp.rfind(" ")
                x = temp[:last_space_index].lower()
                y = temp[last_space_index + 1:]

                # update count(y)
                if y in count:
                    count[y] += 1
                else:
                    count[y] = 1

                # update count(y->x)
                addCount(y, x, emissions)
#   print(count)     #count = {tag1: count1, tag2: count2, etc}
#   print(emissions) #emission = {tag: {word1:count1, word2:count2, etc.}}
    # convert counts to emission probabilities
    for y, xDict in emissions.items():
        for x, xCount in xDict.items():
            xDict[x] = xCount / float(count[y] + k)

        # replace with unk
        emissions[y]["#UNK#"] = k / float(count[y] + k)

    return emissions


def predictSentiments(emissions, testfile, outputfile):

   # predicts sequence labels using argmax(emission)
    # find best #UNK# for later use

    unkTag = "O"
    unkP = 0
    for tag in emissions.keys():
        if emissions[tag]["#UNK#"] > unkP:
            unkTag = tag

    with open(testfile, encoding="utf-8") as f, open(outputfile, "w", encoding="utf-8") as out:
        for line in f:
            if line == "\n":
                out.write(line)
            else:
                word = line.strip().lower()
                # find highest probability for each word
                bestProb = 0
                bestTag = ""
                for tag in emissions:
                    if word in emissions[tag]:
                        if emissions[tag][word] > bestProb:
                            bestProb = emissions[tag][word]
                            bestTag = tag

                if bestTag == "":
                    bestTag = unkTag

                out.write("{} {}\n".format(word, bestTag))
    print("Prediction Done!")


def main(args):
    data = ["EN", "SG", "CN"]
    if args in data:
        dir = Path(args)
        start = time.time()
        emissions = getEmissions(dir/'train')
        predictSentiments(emissions, dir/'dev.in', dir/'dev.p2.out')
        end = time.time()
        print(f"Elapsed time: {round(end-start,2)}s")
    else:
        print("Specified Dataset must be either EN, SG or CN, Run again...")


if __name__ == "__main__":
    args = sys.argv
    main(args[1])
