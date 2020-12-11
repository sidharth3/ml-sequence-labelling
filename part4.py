from part2 import getEmissions
from part3 import getTransitions, convert, isMissing
from pathlib import Path
from math import log
import time, sys


def getTopKViterbi(emissions, transitions, words, textList, k):
  tags = emissions.keys()
  score = {}
  topK_scores = []
  for i in range(k):
    topK_scores.append([0.0,None]); #score with associated parent 


  # forward algorithm
  
  #INITIALISATION PHASE
  score[0] = {"_START": topK_scores}
  # For example : {'B_neutral': [[score1,parent1], [score2, parent2], [score3, parent3]]}

  
  # CONTINUATION PHASE
  for i in range(1, len(textList) + 1):
      word = textList[i - 1].lower()

      # Replace word with #UNK# if not in train
      if word not in words:
          word = "#UNK#"

      for currTag in tags:
          highScores = []
          tempScores = []
          for num in range(k):
            highScores.append([None,None]) #[Score,parent]
            

          # Check that word can be emitted from currTag
          if isMissing(word, currTag, emissions):
              continue

          b = emissions[currTag][word]
          prevHighScore = 0;
          for prevTag, prevScores in score[i - 1].items():

              if isMissing(currTag, prevTag, transitions) :
                  continue

              a = transitions[prevTag][currTag]
              # Calculate score for each Score

              #To loop through the 3 Scores of the previous node and record the highscore 
              #A none score would require no caluclation, hence the loop would be continued on to the next score
              #if the score of any scores are the same, there was also would be no need to caclulate the next highscore twice
              for j in range(k):
                if prevScores[j][0] is None:
                  continue
                if prevScores[j][0] == prevScores[j-1][0]:
                  if j == 0:
                    pass
                  else:
                    continue
                highScore = prevScores[j][0] + log(a) + log(b)
                
                #to look through the current 3 highscores, to make sure that the 3 highscores would be the 3 highest amongst all the differetn calculations
                for p in range(k):
                  
                  if highScores[p][0] is None:
                    highScores[p][0] = highScore;
                    highScores[p][1] = prevTag;
                    break
                  
                  elif highScore == highScores[p-1][0]:
                    break
                  
                  elif highScore > highScores[p][0]:
                    highScores[p][0] = highScore;
                    highScores[p][1] = prevTag;
                    break

          assert(len(highScores)==3)

          #once the 3 high scores are determined, the score for that tag will be replaced with those 3 highscores in the format below
          if i in score:
              score[i][currTag] = highScores
          else:
              score[i] = {currTag: highScores}

  #STOPPING PHASE
  # Final iteration stop case
  highScores = []
  for num in range(k):
    highScores.append([None,None])

  #Loop through all the possible scores in the previous tag
  for prevTag, prevScores in score[len(textList)].items():
      # Check if Prev Tag can lead to a "_STOP"
      if "_STOP" in transitions[prevTag]:
          a = transitions[prevTag]["_STOP"]
          if a == 0 or prevScores[0][0] is None:
              continue

          for j in range(k):
            
            if prevScores[j][0] is None:
              continue
            
            if prevScores[j][0] == prevScores[j-1][0]:
              
              if j == 0:
                pass
              else:
                continue
            
            #No emission value for "_STOP" tag
            highScore = prevScores[j][0] + log(a) 
            
            for p in range(k):
              
              if highScores[p][0] is None:
                highScores[p][0] = highScore;
                highScores[p][1] = prevTag;
                break
              
              elif highScore == highScores[p-1][0]:
                break
              
              elif highScore > highScores[p][0]:
                highScores[p][0] = highScore;
                highScores[p][1] = prevTag;
                break 
  
  #Update "_STOP" tag's score
  score[len(textList) + 1] = {"_STOP": highScores }
  
  # Backtracking Algorithm

  prediction = []
  curr = "_STOP"
  post = ""
  i = len(textList)
  #Initialisation from STOP point 
  
  kScoresAtNode = score[i + 1][curr]
  k_best_score = 10000 #an arbitrary max value
  #Get the 3rd Best score, which is also the smallest of the 3 scores at each Tag
  for x in kScoresAtNode:
    if (x[0] is None):
      if (k_best_score == 10000):
        parent = None
        k_best_score = None
        continue
      else:
        continue
    if x[0] < k_best_score:
      k_best_score = x[0]
  
  #return index of 3rd highest
  if k_best_score != None:
    for index in range(len(kScoresAtNode)):
      if k_best_score in kScoresAtNode[index]:
        final_index = index

    #Get the first chosen parent tag 
    parent = score[i + 1][curr][final_index][1]
  
  if parent is None:
      parent = list(score[i].keys())[0]

  #Append parent to Prediction Array
  prediction.append(parent)
  post = curr
  curr = parent
  i -= 1
  
  #Nth Tag handling

  #At the Nth Tag, there 3 scores to choose from, however one cannot simply choose the third highest at this point
  # This is due to the transition variable that happens from Nth Tag to "_STOP" 
  kScoresAtNode = score[i + 1][curr]
  a = transitions[curr][post]

  #Run through each of the 3 scores at the Nth Tag 
  for s in kScoresAtNode:
    if (s[0] is None):
      parent = None
      continue
    #Sum score with log(a) and if it's equal to the k_best_score then that would be the correct assocaited parent tag
    #K_Best_score would be taken over by the chosen score
    if (s[0] + log(a)== k_best_score):
      parent = s[1]
      k_best_score = s[0]
      break;
  
  if parent is None:
    parent = list(score[i].keys())[0]
  
  #Append Parent to prediction array
  prediction.append(parent)
  post = curr
  curr = parent
  i -= 1
  

  #Continuation to "_START" tag

  while True:
    #Now each word must be taken into consideration as the Emission score must be calculated
    word = textList[i+1].lower()
    
    #If word doesn't exist we'll replace word with UNK
    if word not in words:
      word = "#UNK#"

    #At the ith Tag, there 3 scores to choose from, however one cannot simply choose the third highest at this point
    # This is due to the transition variable and emission variable that happens from ith Tag to (i+1) Tag 
    kScoresAtNode = score[i + 1][curr]
    
    try:
      a = transitions[curr][post]
    except:
      a = transitions["O"]["O"]
    b = emissions[post][word]
    
    #Run through each of the 3 scores at the ith Tag 
    for s in kScoresAtNode:
      if (s[0] is None):
        parent = None
        continue
    #Sum score with log(a) and log(b) and if it's equal to the k_best_score then that would be the correct assocaited parent tag
    #K_Best_score would be taken over by the chosen score
      if (s[0] + log(a) + log(b) == k_best_score):
        parent = s[1]
        k_best_score = s[0]
        break
      parent = None

    if parent is None:
      parent = list(score[i].keys())[0]

    #If parent == "_START", it means it's time to end the backward iteration
    if parent == "_START":
        break


    #Append Parent to prediction array
    prediction.append(parent)
    post = curr
    curr = parent
    i -= 1
  
  #Reverse Prediction as original was from the back to front.  
  prediction.reverse()

  return prediction

def predictWithTopK_Viterbi(emissions, transitions, words, inputFile, outputFile,k):
  with open(inputFile) as f, open(outputFile, "w") as out:
        sentence = []
        for line in f:
            # form sentence
            if line != "\n":
                word = line.strip()
                sentence.append(word)

            # predict tag sequence
            else:
                sequence = getTopKViterbi(emissions, transitions, words, sentence,k)
                #print(sequence)
                for i in range(len(sequence)):
                    out.write("{} {}\n".format(sentence[i], sequence[i]))
                out.write("\n")
                sentence = []
  print("Prediction Done!")


def main(args):
    data = ["EN", "SG", "CN"]
    if args in data:
        dir = Path(args)
        start = time.time()
        emissions = getEmissions(dir/'train')
        transitions = getTransitions(dir/'train')
        vocab = convert(dir/'train')
        predictWithTopK_Viterbi(emissions, transitions, vocab,
                           dir/'dev.in', dir/'dev.p4.out', 3)
        end = time.time()
        print(f"Elapsed time: {round(end-start,2)}s")
    else:
        print("Specified Dataset must be either EN, SG or CN, Run again...")


if __name__ == "__main__":
    args = sys.argv
    main(args[1])