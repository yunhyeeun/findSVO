import nltk, re, random, csv
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Tree
from nltk import ne_chunk
from nltk.tree import ParentedTree
from nltk.tree import MultiParentedTree
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser
from nltk.corpus import wordnet as wn
from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.constituency_parser
import allennlp_models.syntax.biaffine_dependency_parser
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
actionList = ["activate", "activates", "activated",
              "inhibit", "inhibits", "inhibited",
              "bind", "binds",
              "require", "requires", "required",
              "prevent", "prevents", "prevented"]
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
            re.sub(r'\(.*?\)', '', sentence)
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
    CNP: { <NP> <XNP>+ }
    V: { <VBD|VBZ> <VB.*> <IN> | <VB.*>+ | }
    VP: { <V>+ <CNP| NP | PP>* }
    XVP: { <CC|,> <VP> }
    CVP: { <VP> <XVP>+ }
    PP: { <IN|TO> <NP|CNP> }
"""
chunk_parser = RegexpParser(grammar)
constPredictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
dependPredictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

for i in range(100):
    obj = makeObject(dataObject[i])
    putActionList(obj)
dataLists = [activateList, inhibitList, bindList, requireList, preventList]
testList, trainingList = makeTestList(dataLists)
trainingSent = [obj["sentence"] for obj in trainingList]
trainingTriplet = [obj["triplet"] for obj in trainingList]

# EXAMPLE_OBJECT = activateList[0]
# EXAMPLE_SENTENCE = EXAMPLE_OBJECT["sentence"]
# EXAMPLE_TRIPLET = EXAMPLE_OBJECT["triplet"]
# result = predictor.predict(
#   sentence= EXAMPLE_SENTENCE
# )
def find_phrase (tree, phrase):
    leaves = [subtree for subtree in tree if type(subtree) == Tree and subtree.label() == phrase]
    return leaves
# EXAMPLE_TREES = nltk.Tree.fromstring(result["trees"])
# print (find_phrase(EXAMPLE_TREES, "VP"))

f = open ('CS372_HW4_output_20170441.csv', 'w', newline = '', encoding='utf-8-sig')
wr = csv.writer(f)
def findVP(inputList):
    for i in range(20):
        EXAMPLE_OBJECT = inputList[i]
        EXAMPLE_SENTENCE = EXAMPLE_OBJECT["sentence"]
        EXAMPLE_TRIPLET = EXAMPLE_OBJECT["triplet"]
        wr.writerow(["Sentence:{}".format(i)])
        wr.writerow([EXAMPLE_SENTENCE])
        wr.writerow(EXAMPLE_TRIPLET)
        
        # constResult = constPredictor.predict(
        #     sentence = EXAMPLE_SENTENCE
        # )
        dependResult = dependPredictor.predict(
            sentence = EXAMPLE_SENTENCE
        )
        print (dependResult.items())
        # EXAMPLE_CONST_TREES = constResult["trees"]
        # EXAMPLE_DEPEND_TREES = dependResult["trees"]
        # wr.writerow(nltk.Tree.fromstring(EXAMPLE_CONST_TREES))
        # wr.writerow(nltk.Tree.fromstring(EXAMPLE_DEPEND_TREES))
        
        # wr.writerow(find_phrase(nltk.Tree.fromstring(EXAMPLE_TREES), "VP"))
        wr.writerow(["------"])
        break
def makeConstTree(sent):
    constResult = constPredictor.predict(
        sentence = sent
    )
    return ParentedTree.fromstring(constResult["trees"])

def findSubject(t):
    s = []
    for subtree in t.subtrees():
        if subtree.label() == "NP":
            s.append(subtree)
    return s
def checkPassive(t):
    parent = t.parent()
    may_pp = t.right_sibling()
    may_be = parent.left_sibling()
    if may_be != None and may_pp != None:
        if may_be.label().startswith("VB") and may_pp.label() == "PP" and may_pp.leaves()[0] == "by":
            return may_be.leaves()[0] + " " + " ".join(t.leaves()) + " by"
    return " ".join(t.leaves())

def findVerbs(t, phrase, verb):
    for subtree in t:
        if type(subtree) == ParentedTree:
            if subtree.label().startswith("VB") and subtree.leaves()[0] in actionList:
                parent = subtree.parent()
                v = checkPassive(subtree)
                phrase.append(parent)
                verb.append(v)
            findVerbs(subtree, phrase, verb)
    
def findSubSent(t):
    s = []
    for subtree in (t.subtrees()):
        if subtree.label() == "S":
            s.append(subtree)
    return s
def findObject(vps):
    objects = []
    for idx, vp in enumerate (vps):
        newVp = MultiParentedTree.convert(vp)
        for subtree in newVp:
            if subtree.label().startswith("VB") and subtree.leaves()[0] in actionList:
                rightSiblings = subtree.right_siblings()
                objectPhrase = []
                for i, sibling in enumerate (rightSiblings):
                    if rightSiblings[0].label() == "PP":
                        objectPhrase.append(rightSiblings[0])
                        break
                    elif rightSiblings[i].label() == "NP":
                        objectPhrase.append(sibling)
                    elif rightSiblings[i].label() == "CC":
                        continue
                    else:
                        break
                for p in objectPhrase:
                    objects.append(" ".join(p.leaves()))
    return objects
for l in dataLists:
    # findVP(l)
    for obj in l:
        EXAMPLE_SENTENCE = obj["sentence"]
        EXAMPLE_TRIPLET = obj["triplet"]
        wr.writerow([EXAMPLE_SENTENCE])
        wr.writerow(EXAMPLE_TRIPLET)
        constTree = makeConstTree(EXAMPLE_SENTENCE) 
        verbPhrase = []
        verbs = []
        findVerbs(constTree, verbPhrase, verbs)
        objects = findObject(verbPhrase)
        for i in range(len(verbs)):
            wr.writerow([verbs[i]])
            wr.writerow([objects[i]] if i < len(objects) else ["NOT EXIST"])
        # for phrase in verbPhrase:
            # wr.writerow(verbPhrase)
        wr.writerow(["-----"])
    # constTree.draw()
    # verb = findVerb(constTree)
    # for v in actionList:

f.close()




# for SUBTEXT in re.split("that|whether", EXAMPLE_TEXT):
#     SUBTEXT = SUBTEXT.strip()
#     tokens = word_tokenize(SUBTEXT)
#     pos = pos_tag(tokens)
#     newPos = morePos(pos)
#     print (EXAMPLE_TEXT)
#     print (EXAMPLE_TRIPLET)
#     chunk = chunk_parser.parse(newPos)
#     chunk.draw()
# whether: actual-PREPOSITION expected-CONJUNCTION
# 명사, 동사 수일치
# amino: actual-PREPOSITION expected-NOUN
# NOUN PHRASE에서 마지막 noun만 subject or object로 인정(phrase의 head로 선정)
# signaling: actual-VERB expected-NOUN
## 고유명사는 cmudict로 확인할까?
## Subject: ACTION의 왼쪽에서 가장 가까운 형제 NP의 마지막 NN
## Object: ACTION이 포함된 VP의 NP의 마지막 NN
##         ACTION의 오른쪽에서 가장 가까운 형제 NP, PP의 마지막 NN