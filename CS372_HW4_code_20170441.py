import nltk, re, random
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ne_chunk
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
from nltk.corpus import wordnet as wn

f = open(r"input_hw4.txt", "r")
openedFile = f.readlines()
f.close()
text = ""
for line in openedFile:
    text += line
dataObject = text.split("\n---")

activateList = []
inhibitList = []
bindList = []
requireList = []
preventList = []
actionList = ["activate", "activates", "inhibit", "inhibits", "bind", "binds", "require", "requires", "prevent", "prevents"]
preposition = ["in", "on", "of", "at", "with", "by"]
cmudict = nltk.corpus.cmudict.dict()
def putActionList(obj):
    verb = obj["verb"]
    if verb == "activate":
        activateList.append(obj)
    elif verb == "inhibit":
        inhibitList.append(obj)
    elif verb == "bind":
        bindList.append(obj)
    elif verb == "require":
        requireList.append(obj)
    elif verb == "prevent":
        preventList.append(obj)
    else:
        print("CANNOT FIND VERB")

def makeObject(text):
    text = text.strip("\n")
    cgs = text.split("\n")
    verb = ""
    sentence = ""
    triplet = ""
    cited = ""
    for cg in cgs:
        cg = cg.strip()
        if "Verb: " in cg:
            verb = cg.split("Verb: ")[1]
        elif "Sentence: " in cg:
            sentence = cg.split("Sentence: ")[1]
        elif "Triplet: " in cg:
            triplet = cg.split("Triplet: ")[1]
            triplet = triplet.split(", ")
        elif "Cited: " in cg:
            cited = cg.split("Cited: ")[1]
        else:
            break
    return {"verb": verb, "sentence": sentence, "triplet": triplet, "cited": cited}

def makeTestList(lists):
    testidx = random.sample(range(0,20), 4)
    trainingidx = [x for x in range(20) if x not in testidx]
    testList = []
    trainingList = []

    for l in lists:
        testList += [l[x] for x in testidx]
        trainingList += [l[x] for x in trainingidx]
    return testList, trainingList

def wnPos2nltkPos(pos):
    if pos == "n":
        return "NN"
    elif pos == "a" or pos == "s":
        return "JJ"
    elif pos == "v":
        return "VB"
    elif pos == "r":
        return "RB"
    else:
        return None

def morePos(pos):
    newPos = []
    for w, p in pos:
        if w == "," or w == "and" or w == ".":
            newPos.append((w, p))
            continue
        if w == "to":
            newPos.append((w, "TO"))
            continue
        if w in preposition:
            newPos.append((w, "IN"))
            continue
        if cmudict.get(w, "empty") == "empty":
            newPos.append((w, "NN"))
            continue
        synsets = wn.synsets(w)
        if len(synsets) == 0:
            newPos.append((w, p))
            continue
        newP = wnPos2nltkPos(synsets[0].pos())
        if newP != None and p != newP:
            if (newP != "NN"):
                p = newP
        if w in actionList and p.startswith("V") == False:
            newPos.append((w, "VB"))
            continue
        else:
            newPos.append((w, p))
    return newPos

grammar = r"""
    NP: { <DT|PRP>? <JJ.*>* <NN.*>+ }
    XNP: { <CC|,> <NP> }
    CNP: { <NP> <CC|,> <NP>+ }
    V: { <VBD|VBZ> <VBN> <IN> | <VB.*>+ | }
    VP: { <V>+ <CNP| NP | PP>* }
    XVP: { <CC|,> <VP> }
    CVP: { <VP> <XVP>+ }
    PP: { <IN|TO> <Sent|NP|CNP> }
"""
chunk_parser = RegexpParser(grammar)

for i in range(100):
    obj = makeObject(dataObject[i])
    putActionList(obj)

dataLists = [activateList, inhibitList, bindList, requireList, preventList]
testList, trainingList = makeTestList(dataLists)

trainingSent = [obj["sentence"] for obj in trainingList]
trainingTriplet = [obj["triplet"] for obj in trainingList]
EXAMPLE_OBJECT = preventList[4]
EXAMPLE_TEXT = EXAMPLE_OBJECT["sentence"]
EXAMPLE_TRIPLET = EXAMPLE_OBJECT["triplet"]
for SUBTEXT in re.split("that|whether", EXAMPLE_TEXT):
    SUBTEXT = SUBTEXT.strip()
    tokens = word_tokenize(SUBTEXT)
    pos = pos_tag(tokens)
    newPos = morePos(pos)
    print (EXAMPLE_TEXT)
    print (EXAMPLE_TRIPLET)
    chunk = chunk_parser.parse(newPos)
    chunk.draw()
# whether: actual-PREPOSITION expected-CONJUNCTION
# 명사, 동사 수일치
# amino: actual-PREPOSITION expected-NOUN
# NOUN PHRASE에서 마지막 noun만 subject or object로 인정(phrase의 head로 선정)
# signaling: actual-VERB expected-NOUN
## 고유명사는 cmudict로 확인할까?
## Subject: ACTION의 왼쪽에서 가장 가까운 NP의 마지막 NN
## Object: ACTION이 포함된 VP의 NP의 마지막 NN
##         ACTION의 오른쪽에서 가장 가까운 NP, PP의 마지막 NN